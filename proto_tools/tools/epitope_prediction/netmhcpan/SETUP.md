# netMHCpan Setup (gated)

The `netmhcpan-peptides` and `netmhcpan-protein` tools wrap **netMHCpan 4.1**
(developed and verified against **4.1b**), which DTU Health Tech distributes
under an **academic software license** gated behind per-user registration. 

netMHCpan 4.1 cannot be auto-downloaded and the license forbids redistribution,
so in order to use these tools you must accept terms to download the software.

Important:
The license is free for publicly funded academic, educational, and research
institutions. Commercial use requires a separate license from <health-software@dtu.dk>.

## 1. Request the software

1. Go to [services.healthtech.dtu.dk/services/NetMHCpan-4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
   and open the **Downloads** tab, or go straight to the
   [Linux download request form](https://services.healthtech.dtu.dk/cgi-bin/sw_request?software=netMHCpan&version=4.1&packageversion=4.1b&platform=Linux).
2. Read the academic software license agreement on the above page. After agreeing
   to terms of use, you will be able to submit a request to download the software.
   You must specify an institutional e-mail address (@<university>.edu) in order
   to download the software. 
3. If you are approved to access the software, DTU will e-mail you a
  **time-limited download link**. Use it to fetch
   `netMHCpan-4.1b.Linux.tar.gz` (or the macOS build is `netMHCpan-4.1b.Darwin.tar.gz`).

## 2. Extract the software

Once you have downloaded the software from the above link, extract it using the
following command:

```bash
mkdir -p ~/netmhcpan && tar -xzf ~/Downloads/netMHCpan-4.1b.Linux.tar.gz -C ~/netmhcpan
# Produces ~/netmhcpan/netMHCpan-4.1 — this directory is the install root.
```

The archive contains only the software. netMHCpan also needs its network weights
and threshold tables, which DTU publishes as a separate, ungated download. These
are fetched into the install root the first time a netMHCpan tool runs, so there
is nothing further to download by hand.

## 3. Make the install discoverable

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

## 4. Nothing in the install needs editing

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

## 5. Verify

Run this after a netMHCpan tool has been invoked once, so the data files are in
place:

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
- **Data files cannot be fetched automatically.** They are only downloaded into
  an install root that is writable and that identifies itself as 4.1 (the
  `data.tar.gz` URL is version-specific, and 4.1 and 4.2 data are not
  interchangeable). A read-only or unrecognised install root fails fast with the
  reason, and the data can be added by hand:

  ```bash
  curl -L -o /tmp/netmhcpan_data.tar.gz \
    https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz
  tar -xzf /tmp/netmhcpan_data.tar.gz -C ~/netmhcpan/netMHCpan-4.1
  ```

  Extract at the install root, not inside the platform directory: the software
  archive ships `<Platform>_<arch>/data` as a symlink to `../data`, and that
  symlink is the path the tool reads.
- **Unknown allele.** Allele names are checked against the install's
  `MHC_pseudo.dat` before the binary is spawned, so a bad name raises an error
  naming the allele with near-miss suggestions. Note that the binary accepts
  only the unstarred spelling (`HLA-A02:01`); the tools normalize `HLA-A*02:01`
  and `A*02:01` to it for you.
