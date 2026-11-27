# Database Retrieval Tools

Retrieve biological sequences, structures, annotations, and chemical data from major public
databases. These tools wrap public web APIs to fetch records by identifier or query,
establishing provenance and context for downstream analysis. They run on CPU and return
typed, structured records.

- **Input:** an accession, identifier, or search query.
- **Output:** the matching records from public databases, such as sequences, structures, annotations, and chemical data.
- **Provenance:** every output names where its records came from in `source_url`, the request URL with any API key or email removed, next to the record's accession or identifier; the output's `timestamp` records when it was fetched. `ccd-lookup` reads a bundled copy of the CCD offline, and `sequence-fetch` puts `source_url` on each fetched record.
