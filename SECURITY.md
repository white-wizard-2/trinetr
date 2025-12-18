# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. **Email** the maintainers directly (or use GitHub's private vulnerability reporting if available)
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: ASAP (within days)
  - High: Within 2 weeks
  - Medium/Low: Next release cycle

## Security Considerations

### This Project

Trinetr is a visualization tool primarily intended for local development and learning. Key considerations:

- **Model Loading**: Only loads models from trusted sources (torchvision, HuggingFace)
- **File Uploads**: Accepts image files for inference - validate and sanitize
- **Local Execution**: Designed to run locally, not exposed to public internet by default

### If Deploying Publicly

If you choose to deploy this publicly, consider:

1. **Authentication**: Add authentication layer
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Strict file type and size limits
4. **Resource Limits**: Cap memory/CPU usage per request
5. **HTTPS**: Always use TLS in production

## Known Limitations

- No built-in authentication
- No rate limiting
- Designed for trusted local environments
- Large model inference can consume significant resources

## Responsible Disclosure

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities.

