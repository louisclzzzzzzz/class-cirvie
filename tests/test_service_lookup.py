import main as main_module


def test_lookup_known_agent(synthetic_agent_to_service):
    service = main_module.lookup_service("JEAN DUPONT", synthetic_agent_to_service)
    assert service == "SER IND PARIS"


def test_lookup_unknown_agent_returns_none(synthetic_agent_to_service):
    assert main_module.lookup_service("PERSONNE INCONNUE", synthetic_agent_to_service) is None


def test_lookup_inconnu_returns_none(synthetic_agent_to_service):
    assert main_module.lookup_service("INCONNU", synthetic_agent_to_service) is None


def test_lookup_accent_insensitive(synthetic_agent_to_service):
    accented = main_module.lookup_service("ANDRÉ LÉPINE", synthetic_agent_to_service)
    unaccented = main_module.lookup_service("ANDRE LEPINE", synthetic_agent_to_service)
    assert accented == unaccented == "DECES"


def test_service_coverage_metric(cleaned_synthetic_df, synthetic_agent_to_service):
    report = main_module.service_coverage_report(cleaned_synthetic_df, synthetic_agent_to_service)
    assert 0.0 <= report["coverage_pct"] <= 100.0


def test_service_coverage_printed(cleaned_synthetic_df, synthetic_agent_to_service):
    result = main_module.run_pipeline_service(cleaned_synthetic_df, synthetic_agent_to_service)
    assert "agent_to_service" in result
