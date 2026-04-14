// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TLS termination for HTTPS L7 inspection.
//!
//! Provides MITM TLS termination so the proxy can inspect HTTPS traffic.
//! Generates an ephemeral CA at startup, injects it into the sandbox's trust
//! store, terminates TLS from the client (presenting dynamic certs per hostname),
//! inspects the plaintext HTTP, then re-encrypts to upstream using real root CAs.

use miette::{IntoDiagnostic, Result};
use rcgen::{CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, ServerConfig};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// A `TcpStream` wrapper that replays pre-read bytes before delegating to the
/// underlying stream.  Used by the transparent proxy to feed the already-read
/// TLS ClientHello back into the TLS acceptor.
pub struct PrefixedTcpStream {
    prefix: Vec<u8>,
    offset: usize,
    inner: TcpStream,
}

impl PrefixedTcpStream {
    pub fn new(prefix: Vec<u8>, inner: TcpStream) -> Self {
        Self {
            prefix,
            offset: 0,
            inner,
        }
    }
}

impl AsyncRead for PrefixedTcpStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if self.offset < self.prefix.len() {
            let remaining = &self.prefix[self.offset..];
            let to_copy = remaining.len().min(buf.remaining());
            buf.put_slice(&remaining[..to_copy]);
            self.offset += to_copy;
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for PrefixedTcpStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

const MAX_CACHED_CERTS: usize = 256;

/// System CA bundle search paths (common Linux locations).
const SYSTEM_CA_PATHS: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt", // Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",   // RHEL/CentOS/Fedora
    "/etc/ssl/ca-bundle.pem",             // openSUSE
    "/etc/ssl/cert.pem",                  // Alpine/macOS
];

/// Ephemeral CA certificate and key for MITM TLS termination.
#[allow(clippy::struct_field_names)]
pub struct SandboxCa {
    ca_cert: rcgen::Certificate,
    ca_key: KeyPair,
    ca_cert_pem: String,
}

impl SandboxCa {
    /// Generate a new ephemeral CA keypair.
    pub fn generate() -> Result<Self> {
        let ca_key = KeyPair::generate().into_diagnostic()?;

        let mut params = CertificateParams::default();
        params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        params
            .distinguished_name
            .push(DnType::CommonName, "OpenShell Sandbox CA");
        params
            .distinguished_name
            .push(DnType::OrganizationName, "OpenShell");
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];

        let ca_cert = params.self_signed(&ca_key).into_diagnostic()?;
        let ca_cert_pem = ca_cert.pem();

        Ok(Self {
            ca_cert,
            ca_key,
            ca_cert_pem,
        })
    }

    /// Returns the CA certificate in PEM format.
    pub fn cert_pem(&self) -> &str {
        &self.ca_cert_pem
    }
}

/// A leaf certificate chain and private key for a specific hostname.
struct CertifiedLeaf {
    cert_chain: Vec<CertificateDer<'static>>,
    private_key: PrivateKeyDer<'static>,
}

/// Cache of per-hostname leaf certificates signed by the sandbox CA.
pub struct CertCache {
    ca: SandboxCa,
    cache: Mutex<HashMap<String, Arc<CertifiedLeaf>>>,
}

impl CertCache {
    /// Create a new cert cache with the given CA.
    pub fn new(ca: SandboxCa) -> Self {
        Self {
            ca,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or generate a leaf certificate for the given hostname.
    fn get_or_generate(&self, hostname: &str) -> Result<Arc<CertifiedLeaf>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| miette::miette!("cert cache lock poisoned"))?;

        if let Some(leaf) = cache.get(hostname) {
            return Ok(Arc::clone(leaf));
        }

        // Overflow: clear entire map (simple, sufficient for sandbox scale)
        if cache.len() >= MAX_CACHED_CERTS {
            cache.clear();
        }

        let leaf = Arc::new(self.generate_leaf(hostname)?);
        cache.insert(hostname.to_string(), Arc::clone(&leaf));
        Ok(leaf)
    }

    /// Generate a new leaf certificate for the given hostname.
    fn generate_leaf(&self, hostname: &str) -> Result<CertifiedLeaf> {
        let leaf_key = KeyPair::generate().into_diagnostic()?;

        let mut params = CertificateParams::new(vec![hostname.to_string()]).into_diagnostic()?;
        params.distinguished_name.push(DnType::CommonName, hostname);
        params.use_authority_key_identifier_extension = true;

        let leaf_cert = params
            .signed_by(&leaf_key, &self.ca.ca_cert, &self.ca.ca_key)
            .into_diagnostic()?;

        let leaf_der = CertificateDer::from(leaf_cert.der().to_vec());
        let ca_der = CertificateDer::from(self.ca.ca_cert.der().to_vec());
        let key_der = PrivateKeyDer::try_from(leaf_key.serialize_der())
            .map_err(|e| miette::miette!("failed to serialize leaf key: {e}"))?;

        Ok(CertifiedLeaf {
            cert_chain: vec![leaf_der, ca_der],
            private_key: key_der,
        })
    }
}

/// TLS state shared across proxy connections.
pub struct ProxyTlsState {
    cert_cache: CertCache,
    upstream_config: Arc<ClientConfig>,
}

impl ProxyTlsState {
    /// Create a new TLS state with the given cert cache and upstream config.
    pub fn new(cert_cache: CertCache, upstream_config: Arc<ClientConfig>) -> Self {
        Self {
            cert_cache,
            upstream_config,
        }
    }

    /// Get or generate a leaf cert for the hostname and return a TLS acceptor.
    fn acceptor_for(&self, hostname: &str) -> Result<TlsAcceptor> {
        let leaf = self.cert_cache.get_or_generate(hostname)?;
        let mut server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(leaf.cert_chain.clone(), leaf.private_key.clone_key())
            .into_diagnostic()?;
        server_config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(TlsAcceptor::from(Arc::new(server_config)))
    }

    /// Returns a reference to the upstream client config.
    pub fn upstream_config(&self) -> &Arc<ClientConfig> {
        &self.upstream_config
    }
}

/// Accept TLS from a sandbox client, presenting a dynamic cert for the hostname.
///
/// Returns a TLS stream that can be used for plaintext HTTP inspection.
pub async fn tls_terminate_client(
    client: TcpStream,
    tls_state: &ProxyTlsState,
    hostname: &str,
) -> Result<impl AsyncRead + AsyncWrite + Unpin + Send> {
    let acceptor = tls_state.acceptor_for(hostname)?;
    let tls_stream = acceptor.accept(client).await.into_diagnostic()?;
    Ok(tls_stream)
}

/// Accept TLS from a sandbox client where the initial bytes (ClientHello) have
/// already been read.  Wraps the stream with [`PrefixedTcpStream`] so the TLS
/// acceptor sees the complete handshake.
pub async fn tls_terminate_client_prefixed(
    client: TcpStream,
    prefix: Vec<u8>,
    tls_state: &ProxyTlsState,
    hostname: &str,
) -> Result<impl AsyncRead + AsyncWrite + Unpin + Send> {
    let acceptor = tls_state.acceptor_for(hostname)?;
    let prefixed = PrefixedTcpStream::new(prefix, client);
    let tls_stream = acceptor.accept(prefixed).await.into_diagnostic()?;
    Ok(tls_stream)
}

/// Connect TLS to an upstream server, verifying against webpki-roots.
///
/// Returns a TLS stream for re-encrypted upstream communication.
pub async fn tls_connect_upstream(
    upstream: TcpStream,
    hostname: &str,
    client_config: &Arc<ClientConfig>,
) -> Result<impl AsyncRead + AsyncWrite + Unpin + Send> {
    let connector = TlsConnector::from(Arc::clone(client_config));
    let server_name = ServerName::try_from(hostname.to_string()).into_diagnostic()?;
    let tls_stream = connector
        .connect(server_name, upstream)
        .await
        .into_diagnostic()?;
    Ok(tls_stream)
}

/// Build a rustls `ClientConfig` with Mozilla root CAs for upstream connections.
pub fn build_upstream_client_config() -> Arc<ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let mut config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];

    Arc::new(config)
}

/// Write CA certificate files for the sandbox trust store.
///
/// Writes:
/// 1. Standalone CA cert PEM (for `NODE_EXTRA_CA_CERTS` which is additive)
/// 2. Combined bundle: system CAs + sandbox CA (for `SSL_CERT_FILE` which replaces default)
///
/// Returns `(ca_cert_path, combined_bundle_path)`.
pub fn write_ca_files(ca: &SandboxCa, output_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(output_dir).into_diagnostic()?;

    let ca_cert_path = output_dir.join("openshell-ca.pem");
    std::fs::write(&ca_cert_path, ca.cert_pem()).into_diagnostic()?;

    // Read system CA bundle and append our CA
    let mut combined = read_system_ca_bundle();
    if !combined.is_empty() && !combined.ends_with('\n') {
        combined.push('\n');
    }
    combined.push_str(ca.cert_pem());

    let combined_path = output_dir.join("ca-bundle.pem");
    std::fs::write(&combined_path, &combined).into_diagnostic()?;

    Ok((ca_cert_path, combined_path))
}

/// Read the system CA bundle from well-known paths.
fn read_system_ca_bundle() -> String {
    for path in SYSTEM_CA_PATHS {
        if let Ok(contents) = std::fs::read_to_string(path)
            && !contents.is_empty()
        {
            return contents;
        }
    }
    // No system bundle found — combined file will contain only the sandbox CA.
    // This is acceptable since the proxy uses webpki-roots independently.
    String::new()
}

/// Parse PEM certificates from a file into DER-encoded certificates.
pub fn parse_pem_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let file = std::fs::File::open(path).into_diagnostic()?;
    let mut reader = BufReader::new(file);
    rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .into_diagnostic()
}

/// Peek the first bytes of a stream and determine if it looks like a TLS
/// ClientHello handshake.
///
/// A TLS record starts with:
/// - byte 0: `0x16` (ContentType::Handshake)
/// - bytes 1-2: TLS version (0x0301 = TLS 1.0, 0x0302 = TLS 1.1, 0x0303 = TLS 1.2/1.3)
///
/// Returns `true` if the peeked bytes match the TLS handshake pattern.
/// Returns `false` for plaintext HTTP, raw binary, or insufficient data.
pub fn looks_like_tls(peek: &[u8]) -> bool {
    if peek.len() < 3 {
        return false;
    }
    // ContentType::Handshake
    if peek[0] != 0x16 {
        return false;
    }
    // TLS version major must be 0x03 (SSL 3.0 / TLS 1.x)
    if peek[1] != 0x03 {
        return false;
    }
    // TLS version minor: 0x00 (SSL 3.0) through 0x04 (TLS 1.3 record layer)
    peek[2] <= 0x04
}

/// Extract the SNI (Server Name Indication) hostname from a TLS ClientHello.
///
/// Parses the TLS record layer, handshake header, and extensions to find the
/// `server_name` extension (type 0x0000). Returns `None` if the data is not a
/// valid ClientHello or does not contain an SNI extension.
///
/// This is used by the transparent proxy path to determine the intended
/// destination host when a client connects directly (bypassing HTTP CONNECT).
pub fn extract_sni(data: &[u8]) -> Option<String> {
    // Minimum: 5 (record header) + 4 (handshake header) + 2 (version) +
    //          32 (random) + 1 (session_id_len) = 44 bytes
    if data.len() < 44 {
        return None;
    }

    // TLS record: type=0x16, version, length
    if data[0] != 0x16 {
        return None;
    }
    let record_len = u16::from_be_bytes([data[3], data[4]]) as usize;
    let record_end = 5 + record_len;
    if data.len() < record_end {
        return None;
    }

    // Handshake: type=0x01 (ClientHello)
    if data[5] != 0x01 {
        return None;
    }

    // Skip handshake header (4 bytes), client version (2), random (32)
    let mut pos = 5 + 4 + 2 + 32;
    if pos >= record_end {
        return None;
    }

    // Session ID (variable length)
    let session_id_len = data[pos] as usize;
    pos += 1 + session_id_len;
    if pos + 2 > record_end {
        return None;
    }

    // Cipher suites (variable length)
    let cipher_suites_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
    pos += 2 + cipher_suites_len;
    if pos + 1 > record_end {
        return None;
    }

    // Compression methods (variable length)
    let compression_len = data[pos] as usize;
    pos += 1 + compression_len;
    if pos + 2 > record_end {
        return None;
    }

    // Extensions length
    let extensions_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
    pos += 2;
    let extensions_end = pos + extensions_len;
    if extensions_end > record_end {
        return None;
    }

    // Walk extensions looking for server_name (type 0x0000)
    while pos + 4 <= extensions_end {
        let ext_type = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let ext_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if ext_type == 0x0000 && ext_len >= 5 {
            // SNI extension: server_name_list_length (2) + type (1) + name_length (2) + name
            let name_list_end = pos + ext_len;
            if name_list_end > extensions_end {
                return None;
            }
            // Skip server_name_list_length
            let mut sni_pos = pos + 2;
            if sni_pos + 3 > name_list_end {
                return None;
            }
            let name_type = data[sni_pos];
            sni_pos += 1;
            let name_len = u16::from_be_bytes([data[sni_pos], data[sni_pos + 1]]) as usize;
            sni_pos += 2;
            if name_type == 0x00 && sni_pos + name_len <= name_list_end {
                return std::str::from_utf8(&data[sni_pos..sni_pos + name_len])
                    .ok()
                    .map(|s| s.to_string());
            }
            return None;
        }

        pos += ext_len;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ca_generation() {
        let ca = SandboxCa::generate().unwrap();
        let pem = ca.cert_pem();
        assert!(pem.starts_with("-----BEGIN CERTIFICATE-----"));
        assert!(pem.contains("-----END CERTIFICATE-----"));
    }

    #[test]
    fn leaf_cert_generation() {
        let ca = SandboxCa::generate().unwrap();
        let cache = CertCache::new(ca);
        let leaf = cache.get_or_generate("example.com").unwrap();
        assert_eq!(leaf.cert_chain.len(), 2); // leaf + CA
    }

    #[test]
    fn cache_dedup() {
        let ca = SandboxCa::generate().unwrap();
        let cache = CertCache::new(ca);
        let leaf1 = cache.get_or_generate("example.com").unwrap();
        let leaf2 = cache.get_or_generate("example.com").unwrap();
        assert!(Arc::ptr_eq(&leaf1, &leaf2));
    }

    #[test]
    fn cache_overflow_clears() {
        let ca = SandboxCa::generate().unwrap();
        let cache = CertCache::new(ca);

        // Fill cache to capacity
        for i in 0..MAX_CACHED_CERTS {
            cache
                .get_or_generate(&format!("host{i}.example.com"))
                .unwrap();
        }

        // This should trigger a clear and succeed
        let leaf = cache.get_or_generate("overflow.example.com").unwrap();
        assert_eq!(leaf.cert_chain.len(), 2);

        // Cache should now have just one entry
        let cache_inner = cache.cache.lock().unwrap();
        assert_eq!(cache_inner.len(), 1);
    }

    #[test]
    fn looks_like_tls_valid_clienthello() {
        // TLS 1.0 ClientHello
        assert!(looks_like_tls(&[0x16, 0x03, 0x01, 0x00, 0x05]));
        // TLS 1.2
        assert!(looks_like_tls(&[0x16, 0x03, 0x03, 0x01, 0x00]));
        // TLS 1.3 record layer (minor 0x01, but hello advertises 1.3 via extension)
        assert!(looks_like_tls(&[0x16, 0x03, 0x01]));
        // SSL 3.0
        assert!(looks_like_tls(&[0x16, 0x03, 0x00]));
    }

    #[test]
    fn looks_like_tls_rejects_http() {
        assert!(!looks_like_tls(b"GET / HTTP/1.1"));
        assert!(!looks_like_tls(b"POST /api"));
        assert!(!looks_like_tls(b"CONNECT host:443"));
    }

    #[test]
    fn looks_like_tls_rejects_short_input() {
        assert!(!looks_like_tls(&[]));
        assert!(!looks_like_tls(&[0x16]));
        assert!(!looks_like_tls(&[0x16, 0x03]));
    }

    #[test]
    fn looks_like_tls_rejects_non_tls_binary() {
        // SSH protocol
        assert!(!looks_like_tls(b"SSH-2.0-OpenSSH"));
        // Random binary
        assert!(!looks_like_tls(&[0xFF, 0xFE, 0x00]));
        // Wrong content type
        assert!(!looks_like_tls(&[0x17, 0x03, 0x03])); // Application data, not handshake
    }

    #[test]
    fn upstream_config_alpn() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = build_upstream_client_config();
        assert_eq!(config.alpn_protocols, vec![b"http/1.1".to_vec()]);
    }

    #[test]
    fn extract_sni_from_clienthello() {
        // Minimal TLS 1.2 ClientHello with SNI for "gateway.discord.gg"
        let hostname = b"gateway.discord.gg";
        let name_len = hostname.len();

        // Build SNI extension: type(2) + ext_len(2) + list_len(2) + name_type(1) + name_len(2) + name
        let sni_ext_data_len = 2 + 1 + 2 + name_len; // list_len + type + len + name
        let mut sni_ext = vec![
            0x00, 0x00, // extension type: server_name
        ];
        sni_ext.extend_from_slice(&(sni_ext_data_len as u16).to_be_bytes()); // extension data length
        sni_ext.extend_from_slice(&((1 + 2 + name_len) as u16).to_be_bytes()); // server_name_list length
        sni_ext.push(0x00); // name type: host_name
        sni_ext.extend_from_slice(&(name_len as u16).to_be_bytes());
        sni_ext.extend_from_slice(hostname);

        // Extensions block
        let extensions_len = sni_ext.len();
        let mut extensions = Vec::new();
        extensions.extend_from_slice(&(extensions_len as u16).to_be_bytes());
        extensions.extend_from_slice(&sni_ext);

        // ClientHello body: version(2) + random(32) + session_id(1+0) + cipher_suites(2+2) + compression(1+1) + extensions
        let mut client_hello_body = vec![
            0x03, 0x03, // TLS 1.2
        ];
        client_hello_body.extend_from_slice(&[0u8; 32]); // random
        client_hello_body.push(0x00); // session_id length = 0
        client_hello_body.extend_from_slice(&[0x00, 0x02, 0x00, 0xFF]); // cipher suites: 1 suite
        client_hello_body.extend_from_slice(&[0x01, 0x00]); // compression: 1 method (null)
        client_hello_body.extend_from_slice(&extensions);

        // Handshake header: type(1) + length(3)
        let hello_len = client_hello_body.len();
        let mut handshake = vec![0x01]; // ClientHello
        handshake.push(0x00);
        handshake.extend_from_slice(&(hello_len as u16).to_be_bytes());
        handshake.extend_from_slice(&client_hello_body);

        // TLS record header: type(1) + version(2) + length(2)
        let record_len = handshake.len();
        let mut record = vec![
            0x16, // ContentType::Handshake
            0x03, 0x01, // TLS 1.0 record layer
        ];
        record.extend_from_slice(&(record_len as u16).to_be_bytes());
        record.extend_from_slice(&handshake);

        assert_eq!(
            extract_sni(&record),
            Some("gateway.discord.gg".to_string())
        );
    }

    #[test]
    fn extract_sni_returns_none_for_non_tls() {
        assert_eq!(extract_sni(b"GET / HTTP/1.1\r\n"), None);
        assert_eq!(extract_sni(&[]), None);
        assert_eq!(extract_sni(&[0x16, 0x03, 0x01]), None); // too short
    }
}
