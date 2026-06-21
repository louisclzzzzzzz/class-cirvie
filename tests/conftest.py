import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module  # noqa: E402

ORIGINE_TEMPLATES: dict[str, list[str]] = {
    "BASE DE DONNÉES": [
        "Erreur dans la base de données lors de l'extraction des contrats.",
        "Données corrompues détectées dans la base après la mise à jour nocturne.",
        "Incohérence en base de données sur le dossier du sociétaire.",
    ],
    "PROGRAMME": [
        "Bug dans le programme de calcul des capitaux du contrat.",
        "Anomalie applicative détectée dans le module de calcul du contrat.",
        "Le programme de traitement génère un résultat erroné sur le contrat.",
    ],
    "PARAMÉTRAGE": [
        "Mauvais paramétrage de la grille tarifaire du contrat.",
        "Paramétrage incorrect du produit dans le module de gestion du contrat.",
        "Erreur de paramétrage sur les frais appliqués au contrat.",
    ],
    "ASSISTANCE UTILISATEURS": [
        "Demande d'assistance utilisateur pour la prise en main de l'outil CIRVIE sur le contrat.",
        "L'utilisateur sollicite une assistance pour finaliser le dossier contrat.",
        "Assistance utilisateur nécessaire pour comprendre l'anomalie du contrat.",
    ],
    "ERREUR DE SAISIE": [
        "Erreur de saisie manuelle lors de l'enregistrement du dossier contrat.",
        "Faute de saisie sur le montant du capital du contrat.",
        "Erreur de saisie de l'agence sur les informations du sociétaire du contrat.",
    ],
    "DONNÉES": [
        "Données manquantes dans le dossier transmis par l'agence pour le contrat.",
        "Absence de données nécessaires au traitement du dossier sociétaire du contrat.",
        "Données incomplètes reçues pour le contrat du sociétaire.",
    ],
    "BATCH": [
        "Le traitement batch nocturne a échoué impactant le contrat du sociétaire.",
        "Echec du job batch de mise à jour des contrats du sociétaire.",
        "Anomalie batch détectée sur le traitement des capitaux du contrat.",
    ],
    "FISCALITÉ": [
        "Question de fiscalité sur l'imposition des capitaux du contrat décès.",
        "Erreur de calcul fiscal sur l'abattement appliqué au contrat.",
        "Anomalie fiscale détectée sur le prélèvement du contrat du sociétaire.",
    ],
    "RESEAU": [
        "Probleme reseau empechant l'acces au dossier contrat du sociétaire.",
        "Coupure reseau pendant le traitement du dossier contrat du sociétaire.",
        "Lenteur reseau impactant la consultation du contrat du sociétaire.",
    ],
    "TÉLÉPHONE": [
        "Le standard telephonique ne fonctionne plus pour le suivi du contrat.",
        "Probleme telephonique empechant de contacter le sociétaire du contrat.",
        "Anomalie telephonique sur la ligne dediee au contrat du sociétaire.",
    ],
    "MOT DE PASSE": [
        "Le sociétaire ne peut plus valider son contrat car son mot de passe a expiré.",
        "Reinitialisation du mot de passe necessaire pour finaliser le contrat.",
        "Blocage du mot de passe empechant le traitement du dossier contrat.",
    ],
    "HABILITATION": [
        "Probleme d'habilitation empechant la validation du contrat du sociétaire.",
        "Droits d'habilitation manquants pour traiter le dossier contrat.",
        "Anomalie d'habilitation sur l'acces au dossier contrat du sociétaire.",
    ],
    "EDITIQUE": [
        "Le courrier editique du contrat n'a pas ete genere pour le sociétaire.",
        "Anomalie editique sur le document du contrat envoye au sociétaire.",
        "Erreur editique empechant l'impression du contrat du sociétaire.",
    ],
    "INTERFACE": [
        "Probleme d'interface entre les systemes empechant la synchronisation du contrat.",
        "Anomalie d'interface lors du transfert des données du contrat du sociétaire.",
        "Erreur d'interface bloquant la mise a jour du contrat du sociétaire.",
    ],
    "SECURITE": [
        "Alerte de securite detectee sur le dossier contrat du sociétaire.",
        "Anomalie de securite empechant l'acces au contrat du sociétaire.",
        "Probleme de securite identifie sur le traitement du contrat.",
    ],
    "AUTRE": [
        "Anomalie diverse impactant le contrat du sociétaire sans cause identifiee.",
        "Probleme non categorise affectant le dossier contrat du sociétaire.",
        "Incident divers necessitant une intervention sur le contrat du sociétaire.",
    ],
}

ORIGINE_COUNTS: dict[str, int] = {
    "BASE DE DONNÉES": 35,
    "PROGRAMME": 34,
    "PARAMÉTRAGE": 33,
    "ASSISTANCE UTILISATEURS": 33,
    "ERREUR DE SAISIE": 32,
    "DONNÉES": 32,
    "BATCH": 31,
    "FISCALITÉ": 31,
    "RESEAU": 31,
    "TÉLÉPHONE": 30,
    "MOT DE PASSE": 30,
    "HABILITATION": 30,
    "EDITIQUE": 30,
    "INTERFACE": 30,
    "SECURITE": 30,
    "AUTRE": 30,
}

OMV_NON_TEMPLATES: list[str] = [
    "Mot de passe oublié, connexion impossible au poste de travail sans impact contrat.",
    "Le job batch technique est en echec sur le serveur, aucun impact sociétaire.",
    "Ecran bloque au demarrage de l'application, probleme purement technique.",
    "Imprimante hors service dans le service, aucun lien avec un dossier.",
    "Le reseau wifi est indisponible dans les bureaux, probleme infrastructure.",
    "Lenteur generale du poste de travail sans rapport avec un contrat.",
    "Mise a jour logicielle requise sur le poste, aucun impact metier.",
    "Le VPN ne se connecte plus depuis ce matin, probleme reseau interne.",
]

VILLES_SAMPLE = ["PARIS", "NANCY", "LYON", "BORDEAUX", "MARSEILLE", "TOULOUSE", "NANTES", "STRASBOURG", "LILLE", "RENNES"]
CAUSES = ["BASE DE DONNÉES", "PROGRAMME", "PARAMÉTRAGE", "MOT DE PASSE", "BATCH", "RESEAU"]
TRAITES = ["MOA OMVIE", "EXP SUPPORT HABILITATION", "ETD DOMAINE VIE", "EXP OPÉRATIONS DOMAINE VCP"]

AGENTS: list[tuple[str, str, str]] = [
    ("DUPONT", "Jean", "SER IND PARIS"),
    ("MARTIN", "Marie", "DECES"),
    ("BERNARD", "Paul", "SER COLL NANCY"),
    ("THOMAS", "Sophie", "SER IND PARIS"),
    ("PETIT", "Luc", "MOA OMVIE"),
    ("ROBERT", "Claire", "ETD DOMAINE VIE"),
    ("RICHARD", "Marc", "EXP SUPPORT HABILITATION"),
    ("DURAND", "Julie", "DECES"),
    ("LEROY", "Nicolas", "SER COLL NANCY"),
    ("MOREAU", "Camille", "SER IND PARIS"),
    ("SIMON", "Antoine", "EXP OPÉRATIONS DOMAINE VCP"),
    ("LAURENT", "Elodie", "DECES"),
    ("LEFEBVRE", "Hugo", "MOA OMVIE"),
    ("MICHEL", "Lea", "SER IND PARIS"),
    ("GARCIA", "Maxime", "ETD DOMAINE VIE"),
    ("DAVID", "Manon", "SER COLL NANCY"),
    ("BERTRAND", "Thomas", "EXP SUPPORT HABILITATION"),
    ("ROUX", "Chloe", "DECES"),
    ("VINCENT", "Alexandre", "SER IND PARIS"),
    ("LÉPINE", "André", "DECES"),
]

UNKNOWN_NAMES = ["MARTINEZ, Carlos", "NOWAK, Anna", "ROSSI, Marco", "SCHMIDT, Hans"]


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def build_synthetic_agent_to_service() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for last, first, service in AGENTS:
        key = f"{first} {last}".upper()
        mapping[key] = service
        stripped = _strip_accents(key)
        if stripped != key:
            mapping[stripped] = service
    return mapping


def build_synthetic_df(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    def pick_demandeur() -> str:
        choice = rng.random()
        if choice < 0.6:
            last, first, _ = rng.choice(AGENTS)
            return f"{last}, {first}"
        if choice < 0.85:
            return rng.choice(UNKNOWN_NAMES)
        return ""

    def pick_ville() -> str | None:
        return rng.choice(VILLES_SAMPLE) if rng.random() < 0.5 else None

    def pick_urgence() -> str:
        return str(rng.randint(1, 4)) if rng.random() < 0.8 else ""

    for origine, count in ORIGINE_COUNTS.items():
        templates = ORIGINE_TEMPLATES[origine]
        for _ in range(count):
            desc = rng.choice(templates)
            ville = pick_ville()
            if ville:
                desc += f" Sociétaire domicilié à {ville}."
            desc += f" Référence dossier {rng.randint(10000, 99999)}."
            rows.append({
                "Description": desc,
                "INTERVENTION OMV": "OUI",
                "SERVICE": "TOUS",
                "ORIGINE": origine,
                "Traité par": rng.choice(TRAITES),
                "Cause réelle": rng.choice(CAUSES),
                "Urgence": pick_urgence(),
                "Demandeur": pick_demandeur(),
            })

    for _ in range(110):
        desc = rng.choice(OMV_NON_TEMPLATES) + f" Ticket {rng.randint(10000, 99999)}."
        rows.append({
            "Description": desc,
            "INTERVENTION OMV": "NON",
            "SERVICE": "",
            "ORIGINE": "",
            "Traité par": rng.choice(TRAITES),
            "Cause réelle": rng.choice(CAUSES),
            "Urgence": pick_urgence(),
            "Demandeur": pick_demandeur() if rng.random() < 0.3 else "",
        })

    rng.shuffle(rows)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_agent_to_service() -> dict[str, str]:
    return build_synthetic_agent_to_service()


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return build_synthetic_df()


@pytest.fixture(scope="session")
def cleaned_synthetic_df(synthetic_df, synthetic_agent_to_service) -> pd.DataFrame:
    return main_module.clean(synthetic_df, synthetic_agent_to_service)


@pytest.fixture(scope="session")
def trained_omv(cleaned_synthetic_df):
    return main_module.run_pipeline_omv(cleaned_synthetic_df)


@pytest.fixture(scope="session")
def trained_origine(cleaned_synthetic_df):
    return main_module.run_pipeline_origine(cleaned_synthetic_df)


@pytest.fixture(scope="session")
def service_bundle(synthetic_agent_to_service) -> dict:
    return {"agent_to_service": synthetic_agent_to_service, "config": main_module.CONFIG}


@pytest.fixture
def patched_inference_cache(trained_omv, trained_origine, service_bundle, monkeypatch):
    bundles = {"OMV": trained_omv, "ORIGINE": trained_origine, "SERVICE": service_bundle}
    monkeypatch.setattr(main_module, "_INFERENCE_CACHE", bundles)
    return bundles
