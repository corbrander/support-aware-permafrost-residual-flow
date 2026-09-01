from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.nature_article_builder import build_claim_evidence_audit, build_nature_article
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--article-output", default="paper/cold_recon_nature_article.md")
    parser.add_argument("--audit-output", default="paper/cold_recon_claim_evidence_audit.md")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    article = build_nature_article(table_dir, Path(args.article_output))
    audit = build_claim_evidence_audit(table_dir, Path(args.audit_output))
    print(f"nature_article={article}")
    print(f"claim_evidence_audit={audit}")


if __name__ == "__main__":
    main()
