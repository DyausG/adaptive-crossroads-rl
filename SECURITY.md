# Security and scope

This is a local academic simulation, not a traffic-signal deployment system. It has no server, authentication service or database. Do not connect it to road infrastructure or present observed simulation safety as a deployment guarantee.

Install dependencies from the documented sources. Load only trusted model checkpoints. Do not commit tokens, `.env` files, private keys or personal submission credentials. Report vulnerabilities to the repository owner through GitHub private vulnerability reporting when available; otherwise contact the owner privately without posting credentials or exploit-sensitive details in a public issue.

Historical simulator configuration files can contain the original machine's absolute build paths. Runtime input resolution uses the repository location and manifests; those historical paths are provenance, not remote endpoints or credentials.
