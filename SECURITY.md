# Security implementation

| Control | Implementation | Verification |
| --- | --- | --- |
| HTTPS | Render TLS termination; HSTS in production | Public deployment headers |
| Access control | Shared project password, scrypt verification, signed 30-minute session | Login/logout and unauthenticated API tests |
| Secret management | Cloud environment variables; production fails without adequate secrets | Startup configuration check |
| Session protection | Secure in production, HttpOnly, SameSite=Lax | Cookie readback and configuration tests |
| CSRF | Random token in signed session, compared on every POST; rotated on login | Missing/stale token tests |
| Upload controls | CSV/TXT/DAT allowlist, 2 MB request limit, matching WFDB header checks, numeric/finite/duration/rate validation | Invalid and oversized upload tests |
| Privacy | Parse and process in RAM; never persist uploaded ECG files | Code review and API implementation |
| Resource limits | One concurrent inference; 15 POSTs per minute per observed peer; 5 login attempts | Rate-limit test |
| Browser isolation | Same-origin CSP, no external scripts, frame denial, MIME sniffing protection | Header tests |
| Error handling | Generic internal errors; no debug mode or simulated fallback | Invalid-signal tests |
| Model safety | Load only bundled trusted model; uploads cannot become serialized models | Extension allowlist and startup schema check |

## Limits

The password is shared for the demonstration; this is not per-user identity or role-based authorization. Rate limiting is in-memory and process-local and resets on restart. Because proxy forwarding headers are not trusted, limits may apply collectively to visitors behind the same observed proxy peer. This trades concurrency for a bounded classroom deployment; use a shared rate-limit store and validated trusted-proxy configuration before scaling.

Signed cookies prevent tampering but are not encrypted. They contain only authentication state and a CSRF token, never ECG signals. Logs record request paths/status and internal failures, not uploaded ECG contents. Render controls infrastructure TLS and isolation; the app does not claim a clinical compliance certification.

The scikit-learn 1.3.2 pin preserves model serialization compatibility. Upgrading requires re-exporting or retraining the model in the new version and validating predictions. The trusted joblib artifact must not be replaced by unreviewed files. Training performance and current pipeline performance have different provenance.
