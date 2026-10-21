"""Generate three small sample PDFs for a fictional company, so the demo runs without real client data.

    python scripts/make_sample_pdfs.py [output_dir]
"""
import sys
from pathlib import Path

from fpdf import FPDF

FOOTER = "Brightleaf Home is a fictional company. Sample document for a RAG demo."

# file name -> list of pages; each page is a list of (heading, paragraph) pairs
DOCS: dict[str, list[list[tuple[str, str]]]] = {
    "brightleaf-kettle-k2-guide.pdf": [
        [
            ("Brightleaf K2 Smart Kettle", "The K2 is a 1.7 litre electric kettle with five temperature presets: 70, 80, 85, 90 and 100 degrees Celsius. It is rated at 2200 W and boils a full kettle in about 4 minutes."),
            ("Keep Warm Mode", "Press and hold the Warm button for two seconds to keep water at the selected temperature for up to 30 minutes. The ring light turns amber while Keep Warm is active."),
            ("Safety Features", "The kettle switches off automatically when it boils dry or when it is lifted from the base. Never fill above the MAX line, which is 1.7 litres."),
        ],
        [
            ("Descaling", "Descale the K2 every 4 weeks in hard water areas and every 8 weeks in soft water areas. Fill to the 1 litre mark with a mix of half water and half white vinegar, boil once, leave for one hour, then rinse three times with clean water."),
            ("Error Codes", "E1 means the temperature sensor has failed; contact support. E2 means the kettle was switched on empty; let it cool for 10 minutes and refill. E3 means the base is not seated correctly; lift the kettle and place it back on the base."),
            ("App Pairing", "To pair the K2 with the Brightleaf app, hold the Power and Warm buttons together for five seconds until the ring light flashes blue. Pairing works over Bluetooth only; the kettle does not connect to Wi-Fi."),
        ],
    ],
    "brightleaf-returns-and-warranty.pdf": [
        [
            ("1. Returns", "Unused products can be returned within 30 days of delivery for a full refund. Products that have been used can be returned within 14 days if they are faulty. Return shipping is free for faulty items; for change-of-mind returns the customer pays a flat fee of 6.95 EUR."),
            ("2. Refund Timing", "Refunds are issued to the original payment method within 5 business days after the returned item arrives at our warehouse."),
            ("3. Warranty", "All Brightleaf appliances carry a 2 year warranty from the date of purchase. Registering the product in the Brightleaf app within 60 days extends the warranty to 3 years. The warranty does not cover limescale damage caused by skipping descaling."),
            ("4. Making a Claim", "To make a warranty claim, email warranty@brightleaf.example with the order number, a photo of the serial number on the base, and a short description of the fault. Claims are answered within 2 business days."),
        ],
    ],
    "brightleaf-support-faq.pdf": [
        [
            ("Delivery", "Orders placed before 2 pm on a business day ship the same day. Standard delivery within the EU takes 3 to 5 business days. Express delivery takes 1 to 2 business days and costs 9.90 EUR."),
            ("Order Changes", "An order can be changed or cancelled until it is packed, usually within 2 hours of being placed. After that the customer must use the returns process."),
            ("Support Hours", "The support team answers chat and email Monday to Friday from 8:00 to 18:00 Central European Time. Phone support is available for business customers only."),
            ("Spare Parts", "Spare lids, filters and bases for the K2 kettle are sold in the Brightleaf online shop. A replacement limescale filter costs 4.50 EUR."),
        ],
    ],
}


def build(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, pages in DOCS.items():
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        for sections in pages:
            pdf.add_page()
            for heading, text in sections:
                pdf.set_font("Helvetica", "B", 14)
                pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(0, 5, FOOTER, new_x="LMARGIN", new_y="NEXT")
        path = out_dir / name
        pdf.output(str(path))
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/pdfs")
    for p in build(target):
        print(f"wrote {p}")
