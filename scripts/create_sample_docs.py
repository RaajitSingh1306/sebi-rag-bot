"""Generate authentic regulatory sample PDFs for SEBI, RBI, and DPDPA compliance."""
import os
from pathlib import Path
import pymupdf as fitz

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DOCUMENTS = {
    "sebi_lodr_2015.pdf": [
        ("SECURITIES AND EXCHANGE BOARD OF INDIA (LISTING OBLIGATIONS AND DISCLOSURE REQUIREMENTS) REGULATIONS, 2015",
         "CHAPTER I: PRELIMINARY\n"
         "1. Short title and commencement.\n"
         "(1) These regulations may be called the Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations, 2015.\n"
         "(2) They shall apply to listed entities that have listed specified securities, non-convertible debt securities, or other recognized instruments.\n"),
        ("CHAPTER IV: OBLIGATIONS OF LISTED ENTITY",
         "Regulation 23: Related Party Transactions.\n"
         "(1) The listed entity shall formulate a policy on materiality of related party transactions and on dealing with related party transactions.\n"
         "(2) All related party transactions and subsequent material modifications shall require prior approval of the audit committee of the listed entity.\n"
         "(3) Notice period for convening general meeting or audit committee review for related party transactions shall be at least 21 days where required by law.\n"
         "(4) All material related party transactions shall require approval of the shareholders through resolution and no related party shall vote to approve such resolutions.\n"),
        ("CHAPTER IV: CONTINUOUS DISCLOSURES & SHAREHOLDING",
         "Regulation 30: Disclosure of events or information.\n"
         "(1) Every listed entity shall make disclosures of any events or information which, in the opinion of the board of directors of the listed company, are material.\n"
         "(2) Events specified in Para A of Part A of Schedule III are deemed to be material events and the listed entity shall make disclosure of such events within 24 hours of occurrence.\n"
         "\n"
         "Regulation 38: Minimum Public Shareholding.\n"
         "The listed entity shall comply with the minimum public shareholding requirements specified in Rule 19(2) and Rule 19A of the Securities Contracts (Regulation) Rules, 1957 in the manner specified by the Board from time to time.\n"
         "Every listed entity shall maintain public shareholding of at least twenty-five per cent (25%). Any newly listed entity with public shareholding below 25% shall bring it to 25% within a maximum period of three years from the date of listing.\n"),
        ("CHAPTER IV: FINANCIAL RESULTS & GOVERNANCE",
         "Regulation 33: Financial Results.\n"
         "(1) The listed entity shall submit quarterly and year-to-date standalone financial results to the stock exchange within forty-five (45) days of end of each quarter.\n"
         "(2) The annual audited standalone and consolidated financial results shall be submitted within sixty (60) days from the end of the financial year.\n"
         "(3) The audit committee shall review the financial results before submission to the board of directors for approval.\n")
    ],
    "sebi_sast_regulations.pdf": [
        ("SECURITIES AND EXCHANGE BOARD OF INDIA (SUBSTANTIAL ACQUISITION OF SHARES AND TAKEOVERS) REGULATIONS, 2011",
         "CHAPTER I: DEFINITIONS AND OBJECTIVES\n"
         "The SEBI SAST Regulations govern the substantial acquisition of shares, voting rights, and control over target listed companies in India.\n"
         "Acquirer means any person who, directly or indirectly, acquires or agrees to acquire shares or voting rights in the target company, or acquires or agrees to acquire control over the target company, either by himself or through persons acting in concert (PAC).\n"),
        ("CHAPTER II: OPEN OFFER TRIGGERS",
         "Regulation 3: Substantial acquisition of shares or voting rights.\n"
         "(1) No acquirer shall acquire shares or voting rights in a target company which taken together with shares or voting rights, if any, held by him and by persons acting in concert with him in such target company, entitle them to exercise twenty-five per cent (25%) or more of the voting rights in such target company unless the acquirer makes a public announcement of an open offer.\n"
         "(2) Regulation 3(2): Creeping Acquisition Limit.\n"
         "No acquirer who holds twenty-five per cent (25%) or more but less than maximum permissible non-public shareholding (75%) shall acquire additional shares or voting rights entitling him to exercise more than five per cent (5%) of the voting rights in any financial year, unless the acquirer makes a public announcement of an open offer.\n"),
        ("CHAPTER II: ACQUISITION OF CONTROL AND OFFER SIZE",
         "Regulation 4: Acquisition of control.\n"
         "Irrespective of acquisition or holding of shares or voting rights in a target company, no acquirer shall acquire, directly or indirectly, control over such target company unless the acquirer makes a public announcement of an open offer for acquiring shares.\n"
         "\n"
         "Regulation 7: Offer Size.\n"
         "(1) The open offer for acquiring shares to be made by the acquirer and persons acting in concert under regulation 3 and regulation 4 shall be for at least twenty-six per cent (26%) of the total voting share capital of the target company, calculated as of the tenth working day from the closure of the tendering period.\n"),
        ("CHAPTER III: ESCROW ACCOUNT AND PRICING",
         "Regulation 17: Escrow account.\n"
         "(1) Not later than two working days prior to the date of the detailed public statement of the open offer, the acquirer shall create an escrow account toward security for performance of obligations.\n"
         "(2) The escrow account shall be funded for an amount equal to 25% of the first 500 crore rupees of consideration payable under the open offer, and 10% thereafter.\n")
    ],
    "sebi_icdr_2018.pdf": [
        ("SECURITIES AND EXCHANGE BOARD OF INDIA (ISSUE OF CAPITAL AND DISCLOSURE REQUIREMENTS) REGULATIONS, 2018",
         "CHAPTER I: PRELIMINARY\n"
         "These regulations govern initial public offerings (IPO), follow-on public offerings (FPO), rights issues, preferential issues, and bonus issues in India.\n"),
        ("CHAPTER II: INITIAL PUBLIC OFFER ON MAIN BOARD",
         "Regulation 6: Eligibility requirements for an initial public offer.\n"
         "(1) An issuer shall be eligible to make an initial public offer only if:\n"
         "(a) it has net tangible assets of at least three crore rupees in each of the preceding three full years;\n"
         "(b) it has an average operating profit of at least fifteen crore rupees calculated on a restated basis during the three preceding years;\n"
         "(c) it has a net worth of at least one crore rupees in each of the preceding three full years.\n"),
        ("CHAPTER II: PROMOTER CONTRIBUTION AND LOCK-IN",
         "Regulation 14 & 16: Minimum Promoters' Contribution and Lock-in.\n"
         "(1) The promoters of the issuer shall hold at least twenty per cent (20%) of the post-issue capital as minimum promoter contribution.\n"
         "(2) Lock-in period for minimum promoters' contribution:\n"
         "The minimum promoters' contribution of 20% shall be locked in for a period of eighteen (18) months from the date of allotment in the initial public offer (reduced from 3 years for eligible issuers).\n"
         "(3) Promoter holding in excess of 20% shall be locked in for a period of six (6) months from the date of allotment.\n"),
        ("CHAPTER II: ANCHOR INVESTORS AND ALLOTMENT",
         "Regulation 17: Anchor Investor Allocation and Lock-in.\n"
         "(1) Up to 60% of the Qualified Institutional Buyers (QIB) portion may be allocated to anchor investors on a discretionary basis.\n"
         "(2) Lock-in period for anchor investors:\n"
         "For anchor investors, fifty per cent (50%) of the allocated shares shall be locked in for thirty (30) days from the date of allotment, and the remaining fifty per cent (50%) shall be locked in for ninety (90) days from the date of allotment.\n"
         "(3) Minimum subscription requirement for any public offer is ninety per cent (90%) of the offer size.\n")
    ],
    "rbi_model_risk_management.pdf": [
        ("RESERVE BANK OF INDIA: MASTER DIRECTION AND GUIDELINES ON MODEL RISK MANAGEMENT",
         "CHAPTER I: REGULATORY FRAMEWORK\n"
         "1. Model Risk Definition and Scope.\n"
         "Model risk is defined as the potential for adverse consequences resulting from decisions based on incorrect or misused model outputs and reports.\n"
         "All commercial banks, non-banking financial companies (NBFCs), and all-India financial institutions shall establish a comprehensive Model Risk Management (MRM) framework.\n"),
        ("CHAPTER II: MODEL DEVELOPMENT AND VALIDATION",
         "Section 4: Sound Model Development Practices.\n"
         "(1) Financial institutions must rigorously document model design, theoretical rationale, mathematical logic, assumptions, data sources, and sample selection.\n"
         "(2) Machine learning and quantitative trading models must undergo robust feature importance analysis, sensitivity testing, and out-of-time validation before deployment.\n"
         "\n"
         "Section 5: Independent Model Validation.\n"
         "(1) All credit scoring, market risk, and algorithmic models must be independently validated by a qualified team separate from model development.\n"
         "(2) Validation shall evaluate conceptual soundness, ongoing monitoring, outcome analysis, benchmarking against alternative models, and backtesting against historical market regimes.\n"),
        ("CHAPTER III: GOVERNANCE AND AUDIT TRAILS",
         "Section 7: Model Inventory and Annual Review.\n"
         "(1) Regulated entities shall maintain an enterprise-wide model inventory cataloging all active, retired, and under-development models.\n"
         "(2) High-risk models (including capital calculation and automated algorithmic decision systems) shall be reviewed and re-validated at least annually.\n"
         "(3) Complete audit trails of code versioning, parameter updates, training datasets, and performance degradation alerts must be preserved for at least five (5) years.\n")
    ],
    "dpdpa_2023.pdf": [
        ("THE DIGITAL PERSONAL DATA PROTECTION ACT, 2023 (NO. 22 OF 2023)",
         "CHAPTER I: PRELIMINARY AND APPLICABILITY\n"
         "An Act to provide for the processing of digital personal data in a manner that recognizes both the right of individuals to protect their personal data and the need to process such personal data for lawful purposes.\n"
         "This Act applies to the processing of digital personal data within the territory of India where such data is collected in digital form or digitized subsequently.\n"),
        ("CHAPTER II: OBLIGATIONS OF DATA FIDUCIARY",
         "Section 5 & 6: Notice and Consent.\n"
         "(1) Every request for consent shall be preceded or accompanied by a clear notice specifying the personal data sought and the purpose of processing.\n"
         "(2) Consent given by the Data Principal shall be free, specific, informed, unconditional, and unambiguous with a clear affirmative action.\n"
         "(3) The Data Principal shall have the right to withdraw consent at any time with the same ease with which consent was given.\n"),
        ("CHAPTER II: SECURITY SAFEGUARDS AND BREACH REPORTING",
         "Section 8: General obligations of Data Fiduciary.\n"
         "(1) A Data Fiduciary shall implement appropriate technical and organizational measures to ensure effective adherence with the provisions of this Act.\n"
         "(2) A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards to prevent personal data breach.\n"
         "(3) In the event of a personal data breach, the Data Fiduciary shall give the Data Protection Board of India and each affected Data Principal intimation of such breach in such form and manner as may be prescribed.\n"),
        ("CHAPTER VII: PENALTIES AND ADJUDICATION",
         "Section 33: Financial Penalties.\n"
         "The Data Protection Board may impose significant financial penalties for non-compliance:\n"
         "(1) Breach in observing the obligation to take reasonable security safeguards to prevent personal data breach under sub-section (5) of section 8: Penalty may extend up to two hundred and fifty crore rupees (Rs 250,00,00,000 / INR 250 Crores).\n"
         "(2) Failure to notify the Board and affected Data Principals of a personal data breach under sub-section (6) of section 8: Penalty may extend up to two hundred crore rupees (Rs 200,00,00,000).\n"
         "(3) Non-fulfilment of additional obligations in respect of children: Penalty may extend up to two hundred crore rupees.\n")
    ]
}

def create_pdf(filename: str, sections: list):
    pdf_path = DATA_DIR / filename
    doc = fitz.open()
    for title, text in sections:
        page = doc.new_page(width=595, height=842) # A4
        page.insert_text((50, 60), title, fontsize=13, fontname="helv", color=(0.1, 0.2, 0.4))
        # Insert body text
        rect = fitz.Rect(50, 100, 545, 780)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1), lineheight=1.4)
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created {pdf_path} ({len(sections)} pages)")

def main():
    print("Generating regulatory sample PDFs in data/ ...")
    for fname, sections in DOCUMENTS.items():
        create_pdf(fname, sections)
    print("Successfully generated all 5 PDFs.")

if __name__ == "__main__":
    main()
