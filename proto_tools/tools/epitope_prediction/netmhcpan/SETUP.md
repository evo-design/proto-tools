# netMHCpan Setup (gated)

The `netmhcpan-peptides` and `netmhcpan-protein` tools wrap **netMHCpan 4.1**
(developed and verified against **4.1b**), which DTU Health Tech distributes
under an **academic software license** gated behind per-user registration. It
cannot be auto-downloaded and the license forbids redistributing it, so it is
never bundled with proto-tools: you provision it once by hand. The steps below
are copy-paste once you have accepted the terms.

The license is free only for publicly funded academic, education and research
institutions, and it is granted for internal research at a single site.
Commercial use requires a separate license from <health-software@dtu.dk>.

## 1. Accept the license and request the package

1. Go to [services.healthtech.dtu.dk/services/NetMHCpan-4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
   and open the **Downloads** tab, or go straight to the
   [Linux download request form](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux).
2. Read the academic software license agreement shown on that form and **accept
   it explicitly** — you are the licensee, and proto-tools cannot accept the
   terms for you. Then submit the request. An **institutional e-mail address is
   required**: the license is granted only to publicly funded academic,
   education and research institutions, for internal research at a single site.
   Commercial use needs a separate license from <health-software@dtu.dk>.
3. DTU e-mails you a **time-limited download link**. Use it to fetch
   `netMHCpan-4.1b.Linux.tar.gz` (the macOS build is `netMHCpan-4.1b.Darwin.tar.gz`).

## 2. Extract the software

```bash
mkdir -p ~/netmhcpan && tar -xzf ~/Downloads/netMHCpan-4.1b.Linux.tar.gz -C ~/netmhcpan
# Produces ~/netmhcpan/netMHCpan-4.1 — this directory is the install root.
```

## 3. Add the data files (required)

The software archive does **not** contain the network weights and threshold
tables. They are a separate, ungated download that must be extracted *inside*
the `netMHCpan-4.1` directory, where it creates `data/`. netMHCpan cannot run
without it.

```bash
curl -L -o /tmp/netmhcpan_data.tar.gz \
  https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz
tar -xzf /tmp/netmhcpan_data.tar.gz -C ~/netmhcpan/netMHCpan-4.1
# Produces ~/netmhcpan/netMHCpan-4.1/data/. The software archive ships
# <Platform>_<arch>/data as a symlink to it, which is the path the tool reads,
# so extracting at the install root (not inside the platform directory) is what
# makes it resolve.
```

## 4. Make the install discoverable

Either stage it into the managed cache, in which case **no environment variable
is needed**:

```bash
# Resolve the cache location the tool reads from (default ~/.proto/proto_model_cache):
NETMHCPAN_CACHE="${PROTO_MODEL_CACHE:-${PROTO_HOME:-$HOME/.proto}/proto_model_cache}/netmhcpan"
mkdir -p "$NETMHCPAN_CACHE"
cp -a ~/netmhcpan/netMHCpan-4.1/. "$NETMHCPAN_CACHE/"
```

Or keep it where it is and point one variable at the install root:

```bash
export NETMHCPAN_HOME=~/netmhcpan/netMHCpan-4.1
```

`NMHOME` (netMHCpan's own variable) and `PROTO_NETMHCPAN_WEIGHTS_DIR` are
accepted equivalently, and the `netmhcpan_dir` config field overrides all of
them per call. Resolution order is `netmhcpan_dir`, then `NETMHCPAN_HOME`, then
`NMHOME`, then the managed cache.

> Note: if you have set `PROTO_MODEL_CACHE=IN_ENV` or `=NONE`, the
> per-environment cache cannot be pre-populated; use the
> `export NETMHCPAN_HOME=...` alternative above.

## 5. Nothing in the install needs editing

netMHCpan's own `netMHCpan-4.1.readme` tells you to edit the shipped
`netMHCpan` launcher to hardcode `NMHOME` and `TMPDIR`. **Skip that step.**
proto-tools invokes the platform binary
(`<install>/<Platform>_<arch>/bin/netMHCpan`) directly, passing `-rdir` and
`-tdir` and exporting the `NETMHCpan` variable the launcher would have set. So:

- the install stays byte-identical to what DTU shipped, which matters because
  the license prohibits modifying the source without written consent;
- **tcsh is not required** - the shipped launcher is a tcsh script, and it is
  never executed;
- there is no hardcoded path to keep in sync if you move the install.

## 6. Verify

```bash
INSTALL=~/netmhcpan/netMHCpan-4.1          # or "$NETMHCPAN_CACHE"
PLATFORM="$(uname -s)_$(uname -m)"
BASE="$INSTALL/$PLATFORM"
test -x "$BASE/bin/netMHCpan" && test -f "$BASE/data/MHC_pseudo.dat" \
  && cat "$BASE/data/version" \
  && NETMHCpan="$BASE" "$BASE/bin/netMHCpan" -rdir "$BASE" -tdir /tmp/netMHCpanXXXXXX \
       -a HLA-A02:01 -p <(printf 'AAAWYLWEV\n') | grep -E 'NetMHCpan version|AAAWYLWEV '
```

A `NetMHCpan version 4.1b` line followed by a prediction row for `AAAWYLWEV`
(ending in `<= SB`) means both tools will run with no further configuration.
Note that `-tdir` takes an `mkdtemp` template, so the trailing `XXXXXX` is
required; passing a plain directory path fails with `Cannot make tmpdir`.

## Failure modes

- **Nothing configured.** The standalone environment build emits the
  `ASSET_NOT_AVAILABLE` sentinel and exits 64, so env builds and integration
  tests skip cleanly on hosts that have no netMHCpan install.
- **A path is set but wrong.** If `NETMHCPAN_HOME` / `NMHOME` /
  `PROTO_NETMHCPAN_WEIGHTS_DIR` is set and `<Platform>_<arch>/bin/netMHCpan` is
  not there, setup fails immediately with the paths it checked - a typo is an
  error, not a silent skip.
- **Step 3 skipped.** An install whose binary is present but whose
  `data/MHC_pseudo.dat` is missing fails fast, naming `data.tar.gz` and where to
  extract it.
- **Unknown allele.** Allele names are checked against the install's own
  `MHC_pseudo.dat` before the binary is spawned, so a bad name raises an error
  naming the allele with near-miss suggestions. Note that the binary accepts
  only the unstarred spelling (`HLA-A02:01`); the tools normalize `HLA-A*02:01`
  and `A*02:01` to it for you.
