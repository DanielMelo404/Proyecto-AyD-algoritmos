# Proyecto-AyD-algoritmos - Cursor Rules

> Professional development standards for maintaining clean, scalable, and maintainable code.
> **Customize:** Edit the sections marked with `[CUSTOMIZE]` for your project.
> Active Cursor rules live in `.cursor/rules/*.mdc` (generated from this template).

---

## Core Principles

### File Size Limit
**CRITICAL:** Never create files exceeding **500 lines** of code or documentation
- Refactor when approaching 400 lines
- Split into smaller, focused modules
- Each file must have a single, clear responsibility
- Prefer composition over monolithic files


### Folder Structure
Organize code by **feature/domain**, not by technical type.
- Use consistent naming: lowercase with hyphens (kebab-case)
- Keep hierarchies shallow (max 3-4 levels)
- Group related functionality together

---

## Project Structure [CUSTOMIZE]

```
Proyecto-AyD-algoritmos/
├── .cursor/rules/              # Cursor guidance (.mdc)
├── Plan/                       # enunciados, diseño, complejidad
├── algoritmos/                 # implementaciones por tema
│   ├── busqueda/
│   ├── ordenamiento/
│   ├── estructuras/
│   └── grafos/
├── tests/                      # pruebas por algoritmo o problema
├── docs/                       # análisis y notas
├── CLAUDE.md
└── CURSOR_RULES_TEMPLATE.md
```

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
| Type | Convention | Example |
|------|------------|---------|
| Files | kebab-case | `user-profile.tsx` |
| Components/Classes | PascalCase | `UserProfile` |
| Functions/Variables | camelCase | `getUserProfile` |
| Constants | UPPER_SNAKE_CASE | `API_BASE_URL` |
| Python files | snake_case | `user_service.py` |
| Python functions | snake_case | `get_user_profile` |

---

## Language & Framework Standards [CUSTOMIZE]

### JavaScript/TypeScript
- Use strict TypeScript mode
- Define types/interfaces in `types/` directory
- Avoid `any` — use `unknown` if type is truly unknown
- Use type guards for runtime type checking
- Prefer interfaces for object shapes, types for unions/intersections
- [CUSTOMIZE: Add framework-specific standards — React, Vue, Next.js, etc.]

### Python
- Use Python 3.11+ features
- Use type hints for all function parameters and return types
- Use Pydantic for request/response validation
- [CUSTOMIZE: Add framework-specific standards — FastAPI, Django, Flask, etc.]

### [CUSTOMIZE: Add other languages if needed — Go, Rust, etc.]

---

## Database & ORM [CUSTOMIZE]

- **Database:** [CUSTOMIZE: PostgreSQL, MySQL, MongoDB, etc.]
- **ORM/Client:** [CUSTOMIZE: Supabase, Prisma, SQLAlchemy, etc.]
- [CUSTOMIZE: Add table list, RLS policies, migration strategy]

### Database Tables [CUSTOMIZE]

| Table | Purpose |
|-------|---------|
| `[table_name]` | [Description] |
| `[table_name]` | [Description] |

**RLS Policies:** [CUSTOMIZE: Describe RLS strategy or remove if not applicable]

---

## External APIs & Integrations [CUSTOMIZE]

- [CUSTOMIZE: Add AI providers, payment gateways, third-party APIs]
- Store API keys in environment variables
- Handle rate limits with exponential backoff
- Implement proper error handling for API failures

---

## File Organization

### When to Split Files
- File exceeds 400 lines (refactor before hitting 500)
- File contains multiple unrelated concerns
- File has multiple exports that could be grouped differently
- File mixes server and client code (split them)

### File Naming
- Use descriptive names that indicate purpose
- Include file type in name if helpful (e.g., `user.service.ts`, `user-profile.tsx`)
- Group related files in feature folders

### Import Organization
1. External dependencies
2. Internal/absolute imports (`@/` or equivalent alias)
3. Relative imports
4. Type-only imports (`import type`)

---

## API & Data Fetching

- Implement proper error handling and status codes
- Validate input data (Zod, Pydantic, etc.)
- Use environment variables for API URLs
- Document endpoints (OpenAPI/Swagger when applicable)

---

## Error Handling

- Always use try/catch (or equivalent) for async operations
- Provide meaningful error messages
- Log errors with context
- Surface user-friendly messages in UI
- Never expose sensitive information in errors

---

## Security Best Practices

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate and sanitize all user inputs
- Implement proper authentication checks
- Use HTTPS for external requests
- Validate file uploads (MIME types, size limits)

---

## Styling [CUSTOMIZE]

- **Framework:** [CUSTOMIZE: Tailwind, Bootstrap, CSS Modules, etc.]
- **Primary color:** [CUSTOMIZE: e.g., `#4caf50`]
- [CUSTOMIZE: Add design system, component library, theming]

---

## Domain-Specific Architecture — Algoritmos (AyD)

- Organize by algorithm family or problem, not by generic `utils/` dumps.
- Each public algorithm documents precondition, correctness idea, and complexity.
- Tests cover typical, empty, and edge cases; keep benchmarks out of implementation files.

---

## Documentation

- Document complex algorithms and business logic
- Use JSDoc comments for public APIs
- Use Python docstrings for backend functions
- Keep README files updated
- Document environment variables needed
- Comment "why" not "what" (code should be self-explanatory)

---

## Git & Version Control

- Write clear, descriptive commit messages
- Keep commits focused (one logical change per commit)
- Use conventional commit format when possible
- Don't commit generated files or dependencies

---

## [CUSTOMIZE: Docker & Local Development]

- [CUSTOMIZE: Docker setup, docker-compose, hot reload, ports]
- Use `.env.local` for frontend, `.env` for backend
- [CUSTOMIZE: Add shell preference — PowerShell, Bash]

---

## Environment Variables [CUSTOMIZE]

### Frontend
- `[VAR_NAME]` — [Description]

### Backend
- `[VAR_NAME]` — [Description]

Never commit `.env` files. Use `.env.example` as template.

---

## Code Review Checklist

Before submitting code, verify:
- [ ] File is under 500 lines
- [ ] Code follows naming conventions
- [ ] Error handling is implemented
- [ ] No hardcoded values or secrets
- [ ] Code is in correct folders
- [ ] Unused imports/code removed
- [CUSTOMIZE: Add project-specific checklist items]

---

## AI Assistant Guidelines

### Code Generation
- Always respect the 500-line file limit
- Suggest file splits when approaching limit
- Write self-documenting code
- Provide code first, explanation after
- Be concise in explanations

### When Uncertain
- Ask for clarification before making assumptions
- Prefer explicit over implicit
- Document decisions in code comments when non-obvious

---

## Project-Specific Notes [CUSTOMIZE]

### Project Name
- Proyecto-AyD-algoritmos — Análisis y Diseño de algoritmos (implementación + análisis)

### Tech Stack
- **Language:** Python 3.11+ (TypeScript only if the course requires it)
- **Tests:** pytest
- **Database:** none
- **Deployment:** local / academic

### MVP Scope
- Correct implementations, documented complexity, and tests

### Business Rules
- [List critical business rules]

### Reference Documents
- `[path]` — [Description]

---

*Last updated: 2026-09-12 — Edit this file as your project evolves.*
