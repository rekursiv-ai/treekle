# Security Policy

## Why this file exists

treekle decodes plain trees that may come from untrusted sources. With
`allow_imports=True` a tree can name modules to import and callables to call.
Security reports need a private path so exploit details are not
published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- A tree that imports a module or calls code while `allow_imports` is false.
- Denial of service from crafted trees, such as deep nesting or reference
  chains.
- Dependency or packaging issues that affect installed users.
- Supply-chain concerns in the published wheel.
