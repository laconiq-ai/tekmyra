# Security policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Two private channels, either is fine:

- **Email**: <support@tekmyra.ai>
- **GitHub private vulnerability reporting**: the *Report a vulnerability*
  button on this repository's Security tab, which opens a private advisory only
  the maintainers can see

You will get an acknowledgement within **3 working days** and an assessment
within **10 working days**. If you do not hear back in that window, assume the
message did not arrive and try the other channel; a report that silently failed
to reach us is our failure, not yours.

Please include what you need to make it reproducible: the version or commit, the
input, the command, and what happened versus what you expected. If you have a
proof of concept, send it; we will not treat a working demonstration as hostile.

We will tell you what we found, whether we are fixing it, and when. If we
disagree that something is a vulnerability we will say so with our reasoning
rather than letting the report go quiet.

## Scope

This repository is the **open core**: the compression and verification path. It
is a library and a benchmark harness. It ships no server, opens no socket,
authenticates no one, and stores nothing persistently.

**In scope**: anything in this repository, including:

- a way to make the verifier pass an output in which a protected span has been
  dropped, duplicated, or silently altered. This is the security property the
  project exists to enforce, and a bypass of it is the most serious report we
  can receive;
- code execution or file access through fixture loading, model artifact
  loading, or configuration parsing;
- a compression path that leaks span content into an output that was supposed to
  redact or externalize it.

**Out of scope for this repository**: the hosted service, the anchor store, the
tenancy and audit layer, and the economics gate. None of those ship here. If you
have found something in them, the same address reaches us; say so and we will
route it.

**Known and deliberate, so please do not report these as vulnerabilities:**

- **The anchor store resolves nothing.** `NonResolvingAnchorStore.get()` returns
  `None` for every identifier, including ones it minted itself. This is by
  design: the open build cannot vouch for recoverability, so it treats a dropped
  protected span as unrecoverable and refuses. It refuses strictly *more* than
  the commercial build.
- **Minted anchor identifiers carry the tenant and document identifiers in the
  clear** (`[CTX_REF:tenant:doc:chunk:<hash>]`); only the span content is
  hashed. Encryption and keyed identifiers are part of the store, which is not
  in this repository.
- **There is no authentication or authorization anywhere in this code**, because
  there is nothing here to authenticate to.

## Supported versions

Pre-1.0. Security fixes land on `main` and are not backported. Pin a commit if
you need stability.

## Disclosure

Coordinated. We will agree a date with you, and we will credit you unless you
ask us not to. If we have not shipped a fix within 90 days of your report, you
are free to publish; we would rather you publish than sit on it indefinitely
because we were slow.
