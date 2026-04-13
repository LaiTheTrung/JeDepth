# Commit Rules

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification to ensure a consistent, readable, and automatable commit history.

---

## Commit Message Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

| Part        | Required | Description                                                                 |
| ----------- | :------: | --------------------------------------------------------------------------- |
| **type**    |    ✅    | The category of the change (see [Types](#types) below).                     |
| **scope**   |    ❌    | A noun describing the section of the codebase affected (e.g., `planner`, `depth`, `docker`). |
| **subject** |    ✅    | A short, imperative description of the change (≤ 72 characters).            |
| **body**    |    ❌    | A detailed explanation of *what* and *why* (not *how*).                      |
| **footer**  |    ❌    | References to issues, breaking changes, etc.                                |

---

## Types

| Type         | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `feat`       | A new feature or capability.                                       |
| `fix`        | A bug fix.                                                         |
| `docs`       | Documentation-only changes.                                        |
| `style`      | Code style changes (formatting, whitespace) — **no logic change**. |
| `refactor`   | Code restructuring without changing external behavior.             |
| `perf`       | Performance improvements.                                          |
| `test`       | Adding or updating tests.                                          |
| `build`      | Changes to the build system or external dependencies.              |
| `ci`         | Changes to CI/CD configuration files and scripts.                  |
| `chore`      | Maintenance tasks that don't modify `src` or `test` files.         |
| `revert`     | Reverts a previous commit.                                         |

---

## Rules

1. **Use the imperative mood** in the subject line (e.g., *"add feature"*, not *"added feature"*).
2. **Do not** end the subject line with a period.
3. **Limit the subject line** to 72 characters.
4. **Separate** the subject from the body with a blank line.
5. **Use the body** to explain *what* and *why*, not *how*.
6. **Mark breaking changes** with a `BREAKING CHANGE:` footer or a `!` after the type/scope.

---

## Examples

### Simple commit
```
feat(planner): add dynamic obstacle re-routing
```

### Commit with body
```
fix(depth): correct stereo rectification offset

The horizontal offset was calculated using the wrong baseline value,
causing depth maps to drift at long range.
```

### Breaking change
```
refactor(bridge)!: rename PX4 topic from /odom to /vehicle_odometry

BREAKING CHANGE: All subscribers to the old /odom topic must update
their configuration to use /vehicle_odometry.
```

### Chore / maintenance
```
chore(docker): bump CUDA base image to 12.4
```

---

## Branch Naming (Recommended)

Use a similar convention for branch names:

```
<type>/<short-description>
```

**Examples:**
- `feat/stereo-depth-model`
- `fix/planner-collision-check`
- `docs/update-commit-rules`

---

> **Tip:** Use `git log --oneline` to verify your commit history stays clean and readable.