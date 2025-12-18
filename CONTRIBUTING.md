# Contributing to Trinetr

Thanks for your interest in contributing! Here's how you can help.

## Ways to Contribute

- **Bug Reports**: Found something broken? Open an issue with steps to reproduce.
- **Feature Requests**: Have an idea? Open an issue to discuss.
- **Code**: Fix bugs or implement features via pull requests.
- **Documentation**: Improve docs, add examples, fix typos.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/trinetr.git
cd trinetr

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup (in another terminal)
cd frontend
npm install
npm run dev
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** and test locally
4. **Commit** with clear messages: `git commit -m "Add: feature description"`
5. **Push**: `git push origin feature/your-feature-name`
6. **Open PR** against `main` branch

## Code Guidelines

### Python (Backend)
- Follow PEP 8 style
- Add type hints where possible
- Keep functions focused and small
- Don't break existing API endpoints

### TypeScript (Frontend)
- Use functional components with hooks
- Follow existing component patterns
- Keep components focused
- Use TypeScript types properly

## Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: what was changed`
- `Remove: what was removed`
- `Refactor: what was refactored`

## Testing

Before submitting:
1. Ensure the backend starts without errors
2. Ensure the frontend builds successfully
3. Test your changes with both CNN and Transformer models
4. Check browser console for errors

## Questions?

Open an issue with the `question` label or start a discussion.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

