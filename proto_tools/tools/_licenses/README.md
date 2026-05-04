# `_licenses/` — Shared license metadata

This directory holds canonical text for every SPDX-allowlisted license used by
the toolkits in `proto_tools/tools/`. One file per license (`Apache-2.0.txt`,
`CC-BY-4.0.txt`, etc.); each toolkit's `license.yaml` references the SPDX ID
and the resolver pulls the text from here. Custom vendor terms live inline
in the toolkit's `license.yaml` instead.

## Field reference for `license.yaml`

The schema is enforced by `tests/style_consistency_tests/test_license_consistency.py`.
Three of the operational fields are easy to confuse — read them with the
following meanings:

### `redistribution: bool`

**Question answered:** Can this tool be offered as a hosted service?

- `true`  — the tool's license permits it to be packaged and served via a
  hosted API; you can choose between running it locally and calling a hosted
  version.
- `false` — the tool's license forbids it from being included in a hosted
  service. To use it, you must run it yourself with your own weights and
  credentials. Typical reasons: non-commercial weights (CC-BY-NC-*) or
  vendor terms that prohibit redistribution.

### `commercial_use: yes | no | restricted`

**Question answered:** May you use the tool's outputs commercially?

- `yes` — outputs may be used in any commercial context (Apache-2.0, MIT,
  CC-BY-4.0).
- `no` — outputs may not be used commercially (CC-BY-NC-* and similar
  non-commercial licenses).
- `restricted` — commercial use requires a separate contract or license with
  the upstream vendor (some custom proprietary terms).

### `attribution_required: bool`

**Question answered:** Must you display attribution to the upstream creator
when using the tool or its outputs?

CC-BY-* and CC-BY-NC-* licenses → `true`. Apache-2.0 / MIT typically don't
require runtime attribution → `false` (NOTICE preservation only applies if
you redistribute source, which is rarely the wrapper's concern).

## Quick decision matrix

| License (weights) | redistribution | commercial_use | attribution_required |
|---|---|---|---|
| Apache-2.0 | true | yes | false |
| MIT | true | yes | false |
| CC-BY-4.0 | true | yes | true |
| CC-BY-NC-4.0 | false | no | true |
| CC-BY-NC-SA-4.0 | false | no | true |
| `Custom (...)` non-commercial vendor terms | usually false | no | depends |

When code and weights have different licenses, the **most restrictive** of
the two drives `redistribution` and `commercial_use`.

## Adding a new SPDX license

1. Drop the canonical license text into `_licenses/{spdx-id}.txt` (use the
   SPDX-canonical filename, e.g. `BSD-3-Clause.txt`).
2. Add the SPDX identifier to `_ALLOWED_SPDX` in
   `tests/style_consistency_tests/test_license_consistency.py`.
3. Reference it from any `license.yaml` via `spdx: <id>` — do not inline the
   text again.
