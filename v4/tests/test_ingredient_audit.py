from scripts.audit_ingredients import (
    IngredientAuditRow,
    build_alias_gap_report,
    render_markdown_report,
)


def test_alias_gap_report_detects_missing_db_and_code_aliases() -> None:
    rows = [
        IngredientAuditRow(
            slug="concombre",
            name="Concombre",
            aliases=["khyar"],
            category="legume",
            article_count=3,
            ingredients_list_mentions=5,
            example_articles=["122: Fattouche"],
        ),
        IngredientAuditRow(
            slug="menthe",
            name="Menthe",
            aliases=[],
            category="herbe",
            article_count=2,
            ingredients_list_mentions=4,
            example_articles=["456: Taboulé"],
        ),
    ]

    report = build_alias_gap_report(
        rows,
        code_aliases={
            "concombre": ("concombre", "concombres", "khyar", "khiar"),
            "menthe": ("menthe", "menthes", "naana"),
            "tomate": ("tomate",),
        },
    )

    assert report.ingredients_without_db_aliases == ["menthe"]
    assert report.code_aliases_missing_in_db["concombre"] == ["concombres", "khiar"]
    assert report.code_aliases_missing_in_db["menthe"] == ["menthes", "naana"]
    assert "tomate" not in report.code_aliases_missing_in_db


def test_markdown_report_includes_summary_rows_and_gaps() -> None:
    rows = [
        IngredientAuditRow(
            slug="concombre",
            name="Concombre",
            aliases=["khyar"],
            category="legume",
            article_count=3,
            ingredients_list_mentions=5,
            example_articles=["122: Fattouche"],
        ),
    ]
    gaps = build_alias_gap_report(
        rows,
        code_aliases={"concombre": ("concombre", "concombres", "khyar")},
    )

    markdown = render_markdown_report(rows, gaps)

    assert "# Audit ingredients Sahtein v4" in markdown
    assert "| concombre | Concombre | legume | 3 | 5 | khyar |" in markdown
    assert "`concombre`: concombres" in markdown
