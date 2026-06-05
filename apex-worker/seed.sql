-- APEX Research Agent — Seed Data
-- Source tier rules from source_tiers.yaml

-- P1: Primary — peer-reviewed, institutional, or officially published
INSERT OR IGNORE INTO source_tier_rules (id, domain_pattern, tier, doc_types, boost_factor, max_age_days) VALUES
    ('p1-arxiv', 'arxiv.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-pubmed', 'pubmed.ncbi.nlm.nih.gov', 'P1', '["paper","dataset"]', 1.5, NULL),
    ('p1-nature', 'nature.com', 'P1', '["paper","article"]', 1.5, NULL),
    ('p1-science', 'science.org', 'P1', '["paper","article"]', 1.5, NULL),
    ('p1-nejm', 'nejm.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-lancet', 'lancet.com', 'P1', '["paper"]', 1.5, NULL),
    ('p1-acm', 'dl.acm.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-ieee', 'ieee.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-springer', 'springer.com', 'P1', '["paper","book"]', 1.5, NULL),
    ('p1-wiley', 'wiley.com', 'P1', '["paper","book"]', 1.5, NULL),
    ('p1-semantic', 'semanticscholar.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-openreview', 'openreview.net', 'P1', '["paper"]', 1.5, NULL),
    ('p1-biorxiv', 'biorxiv.org', 'P1', '["paper"]', 1.5, NULL),
    ('p1-medrxiv', 'medrxiv.org', 'P1', '["paper"]', 1.5, NULL);

-- P2: Secondary — institutional, government, or reputable organization
INSERT OR IGNORE INTO source_tier_rules (id, domain_pattern, tier, doc_types, boost_factor, max_age_days) VALUES
    ('p2-edu', '*.edu', 'P2', '["paper","article","report"]', 1.2, 365),
    ('p2-acuk', '*.ac.uk', 'P2', '["paper","article"]', 1.2, 365),
    ('p2-acjp', '*.ac.jp', 'P2', '["paper"]', 1.2, 365),
    ('p2-nih', 'nih.gov', 'P2', '["paper","report"]', 1.2, 365),
    ('p2-nasa', 'nasa.gov', 'P2', '["paper","dataset","report"]', 1.2, 365),
    ('p2-cdc', 'cdc.gov', 'P2', '["report","legal"]', 1.2, 365),
    ('p2-who', 'who.int', 'P2', '["report","legal"]', 1.2, 365),
    ('p2-nist', 'nist.gov', 'P2', '["paper","report"]', 1.2, 365),
    ('p2-govuk', 'gov.uk', 'P2', '["legal","report"]', 1.2, 365);

-- P3: Tertiary — community, crowdsourced, or blog content
INSERT OR IGNORE INTO source_tier_rules (id, domain_pattern, tier, doc_types, boost_factor, max_age_days) VALUES
    ('p3-medium', 'medium.com', 'P3', '["article"]', 1.0, 180),
    ('p3-substack', 'substack.com', 'P3', '["article"]', 1.0, 180),
    ('p3-wikipedia', 'wikipedia.org', 'P3', '["article"]', 1.0, 180),
    ('p3-stackoverflow', 'stackoverflow.com', 'P3', '["article"]', 1.0, 180),
    ('p3-reddit', 'reddit.com', 'P3', '["article"]', 1.0, 180);

-- UNV: Unverified — catch-all
INSERT OR IGNORE INTO source_tier_rules (id, domain_pattern, tier, doc_types, boost_factor, max_age_days) VALUES
    ('unv-wildcard', '*', 'UNV', '["other"]', 0.8, 90);
