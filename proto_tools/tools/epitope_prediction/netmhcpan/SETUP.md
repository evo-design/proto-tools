# netMHCpan Setup (gated)

The `netmhcpan-peptides` and `netmhcpan-protein` tools wrap **netMHCpan 4.2**
(developed and verified against **4.2e**), which DTU Health Tech distributes
under an **academic software license** gated behind per-user registration.

netMHCpan 4.2 cannot be auto-downloaded and the license forbids redistribution,
so in order to use these tools you must accept terms to download the software.

Important:
The license is free for publicly funded academic, educational, and research
institutions. Commercial use requires a separate license from <health-software@dtu.dk>.

## 1. Request the software

1. Go to [services.healthtech.dtu.dk/services/NetMHCpan-4.2](https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/)
   and open the **Downloads** tab, or go straight to the
   [Linux download request form](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.2&packageversion=4.2estatic&platform=Linux).
   That form requests the statically linked Linux build, which is the one to
   prefer on older distributions; dynamically linked Linux, `Linux_arm64`,
   `Darwin_x86_64` and `Darwin_arm64` builds are listed on the Downloads tab.
2. Read the academic software license agreement on the above page. After agreeing
   to terms of use, you will be able to submit a request to download the software.
   You must specify an institutional e-mail address (@<university>.edu) in order
   to download the software.
3. If you are approved to access the software, DTU will e-mail you a
   **time-limited download link** (4 hours). Use it to fetch
   `netMHCpan-4.2estatic.Linux.tar.gz` (~116 MB) along with
   `netMHCpan-4.2_license.txt`. Keep the archive: the link expires, and it is the
   only copy you are licensed to hold.

## 2. Extract the software

Once you have downloaded the software from the above link, extract it using the
following command:

```bash
mkdir -p ~/netmhcpan && tar -xzf ~/Downloads/netMHCpan-4.2estatic.Linux.tar.gz -C ~/netmhcpan
# Produces ~/netmhcpan/netMHCpan-4.2 — this directory is the install root.
```

The archive **bundles the network weights and threshold tables**, so there is no
separate data download and nothing is fetched on first run.

## 3. Make the install discoverable

Either stage it into the managed cache, in which case **no environment variable
is needed**:

```bash
# Resolve the cache location the tool reads from (default ~/.proto/proto_model_cache):
NETMHCPAN_CACHE="${PROTO_MODEL_CACHE:-${PROTO_HOME:-$HOME/.proto}/proto_model_cache}/netmhcpan"
mkdir -p "$NETMHCPAN_CACHE"
cp -a ~/netmhcpan/netMHCpan-4.2/. "$NETMHCPAN_CACHE/"
```

Or keep it where it is and point one variable at the install root:

```bash
export NETMHCPAN_HOME=~/netmhcpan/netMHCpan-4.2
```

`NMHOME` (a variable of the netMHCpan software) and `PROTO_NETMHCPAN_WEIGHTS_DIR` are
accepted equivalently, and the `netmhcpan_dir` config field overrides all of
them per call. Resolution order is `netmhcpan_dir`, then `NETMHCPAN_HOME`, then
`NMHOME`, then the managed cache.

> Note: if you have set `PROTO_MODEL_CACHE=IN_ENV` or `=NONE`, the
> per-environment cache cannot be pre-populated; use the
> `export NETMHCPAN_HOME=...` alternative above.

## 4. Nothing in the install needs editing

The `netMHCpan-4.2.readme` shipped with the software tells you to edit the shipped
`netMHCpan` launcher to hardcode `NMHOME` and `TMPDIR`. **Skip that step.**
proto-tools invokes the platform binary
(`<install>/<Platform>_<arch>/bin/netMHCpan-4.2`) directly, passing `-rdir` and
`-tdir` and exporting the `NETMHCpan` variable the launcher would have set. So:

- the install stays byte-identical to what DTU shipped, which matters because
  the license prohibits modifying the source without written consent;
- **tcsh is not required** - the shipped launcher is a tcsh script, and it is
  never executed;
- there is no hardcoded path to keep in sync if you move the install.

## 5. Verify

```bash
INSTALL=~/netmhcpan/netMHCpan-4.2          # or "$NETMHCPAN_CACHE"
PLATFORM="$(uname -s)_$(uname -m)"
BASE="$INSTALL/$PLATFORM"
test -x "$BASE/bin/netMHCpan-4.2" && test -f "$BASE/data/MHC_pseudo.dat" \
  && cat "$BASE/data/version" \
  && NETMHCpan="$BASE" "$BASE/bin/netMHCpan-4.2" -rdir "$BASE" -tdir /tmp/netMHCpanXXXXXX \
       -a HLA-A02:01 -p <(printf 'AAAWYLWEV\n') | grep -E 'NetMHCpan version|AAAWYLWEV '
```

A `NetMHCpan version 4.2e` line followed by a prediction row for `AAAWYLWEV`
(ending in `<= WB`) means both tools will run with no further configuration.
Note that `-tdir` takes an `mkdtemp` template, so the trailing `XXXXXX` is
required; passing a plain directory path fails with `Cannot make tmpdir`.

## Failure modes

- **Nothing configured.** The standalone environment build emits the
  `ASSET_NOT_AVAILABLE` sentinel and exits 64, so env builds and integration
  tests skip cleanly on hosts that have no netMHCpan install.
- **A path is set but wrong.** If `NETMHCPAN_HOME` / `NMHOME` /
  `PROTO_NETMHCPAN_WEIGHTS_DIR` is set and `<Platform>_<arch>/bin/netMHCpan-4.2` is
  not there, setup fails immediately with the paths it checked - a typo is an
  error rather than a silent skip.
- **Data files missing.** Because the archive ships them, an install whose
  `<Platform>_<arch>/data` does not resolve is a partial extraction. Re-extract
  the archive at the install root; the platform directory ships
  `<Platform>_<arch>/data` as a symlink to `../data`, and that symlink is the
  path the tool reads.
- **Unknown allele.** Allele names are checked against the install's
  `allelenames` list before the binary is spawned, so a bad name raises an error
  naming the allele with near-miss suggestions. Note that the binary accepts
  only the unstarred spelling (`HLA-A02:01`); the tools normalize `HLA-A*02:01`
  and `A*02:01` to it for you.
