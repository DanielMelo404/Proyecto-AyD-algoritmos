# CLAUDE.md

> Professional development standards for maintaining clean, scalable, and maintainable code.
> These guidelines are general-purpose and apply to any project, in any language or framework.

---

## Core Principles

### File Size Limit
**CRITICAL:** Avoid creating files exceeding **500 lines** of code or documentation.
- Refactor when approaching 400 lines
- Split into smaller, focused modules
- Each file must have a single, clear responsibility
- Prefer composition over monolithic files

### Folder Structure
Organize code by **feature/domain**, not by technical type.
- Use consistent naming (kebab-case is a safe default)
- Keep hierarchies shallow (max 3–4 levels)
- Group related functionality together

---

## Code Quality Standards

### DRY (Don't Repeat Yourself)
- Extract repeated logic into reusable functions/components
- Create shared utilities for common operations
- Use constants for magic numbers and strings

### KISS (Keep It Simple, Stupid)
- Prefer simple, readable solutions
- Avoid premature optimization
- Write code a junior developer can understand

### Single Responsibility Principle
- Each function/component does one thing well
- Split functions that do multiple things
- Modules should have a single reason to change

### Naming Conventions
Match the conventions already used in the codebase. When starting fresh, these are sensible defaults:

| Type | Convention | Example |
|------|------------|---------|
| Files (JS/TS) | kebab-case | `user-profile.tsx` |
| Components/Classes | PascalCase | `UserProfile` |
| Functions/Variables | camelCase | `getUserProfile` |
| Constants | UPPER_SNAKE_CASE | `API_BASE_URL` |
| Python files/functions | snake_case | `user_service.py`, `get_user_profile` |

---

## Language & Framework Standards

- Follow the idioms and style already present in the project — consistency beats personal preference.
- Match existing formatting, linting, and import ordering rules (respect `.editorconfig`, ESLint, Prettier, Black, etc.).

### Typed Languages (TypeScript, etc.)
- Enable and respect strict mode
- Avoid `any` — use `unknown` when a type is truly unknown, then narrow with type guards
- Prefer interfaces for object shapes, types for unions/intersections
- Keep shared type definitions in a dedicated location

### Dynamic Languages (Python, etc.)
- Use type hints for function parameters and return values
- Validate inputs at boundaries (e.g. Pydantic, Zod, or equivalent)
- Follow the language's standard style guide (PEP 8 for Python)

---

## File Organization

### When to Split Files
- File exceeds 400 lines (refactor before hitting 500)
- File contains multiple unrelated concerns
- File has multiple exports that could be grouped differently
- File mixes server and client code (split them)

### File Naming
- Use descriptive names that indicate purpose
- Include a type suffix when it aids clarity (e.g. `user.service.ts`, `user-profile.tsx`)
- Group related files in feature folders

### Import Organization
1. External dependencies
2. Internal/absolute imports (`@/` or equivalent alias)
3. Relative imports
4. Type-only imports (`import type`)

---

## API & Data Fetching

- Implement proper error handling and meaningful status codes
- Validate input data at the boundary
- Use environment variables for API URLs and configuration
- Document endpoints (OpenAPI/Swagger) when applicable

---

## Error Handling

- Always use try/catch (or the language equivalent) for operations that can fail
- Provide meaningful error messages
- Log errors with enough context to debug
- Surface user-friendly messages in the UI
- Never expose sensitive information in errors

---

## Security Best Practices

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate and sanitize all user inputs
- Implement proper authentication and authorization checks
- Use HTTPS for external requests
- Validate file uploads (MIME types, size limits)
- Handle third-party rate limits with retries/exponential backoff

---

## Documentation

- Document complex algorithms and business logic
- Use doc comments (JSDoc, docstrings, etc.) for public APIs
- Keep README files updated
- Document required environment variables
- Comment "why," not "what" — code should be self-explanatory

---

## Git & Version Control

- Write clear, descriptive commit messages
- Keep commits focused (one logical change per commit)
- Use conventional commit format when possible
- Don't commit generated files, dependencies, or `.env` files
- Provide a `.env.example` as a template for required variables

---

## Code Review Checklist

Before submitting code, verify:
- [ ] File is under 500 lines
- [ ] Code follows the project's naming and style conventions
- [ ] Error handling is implemented
- [ ] No hardcoded values or secrets
- [ ] Code is in the correct folders
- [ ] Unused imports/code removed
- [ ] Tests pass and cover new behavior

---

## AI Assistant Guidelines

### Working in the Codebase
- Read surrounding code before editing; match its style, patterns, and conventions
- Respect the 500-line file limit and suggest splits when approaching it
- Prefer editing existing files over creating new ones unless a new module is warranted
- Write self-documenting code; add comments only where intent is non-obvious
- Make the smallest change that fully solves the problem

### Communication
- Provide code first, concise explanation after
- Be direct and avoid unnecessary preamble

### When Uncertain
- Ask for clarification before making assumptions
- Prefer explicit over implicit
- Verify facts against the codebase rather than guessing
- Don't invent APIs, files, or configuration that you haven't confirmed exist

---

*Adapt and extend this file as the project evolves.*
