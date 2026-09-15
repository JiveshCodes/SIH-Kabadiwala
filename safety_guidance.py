"""
======================================================================
KABADIWALA ML PIPELINE: VERNACULAR SAFETY GUIDANCE ENGINE (SIH #26229)
======================================================================
Provides audio-visual safety instructions in Hindi, Marathi, and English
to prevent hazardous informal recycling practices (cable burning, acid leaching,
CRT smashing, battery punctures).
"""

SAFETY_GUIDANCE_CATALOG = {
    "cables_wires": {
        "hazard_title": {
            "hi": "तारों को खुले में न जलाएं (Open Wire Burning Hazard)",
            "mr": "वायरी उघड्यावर जाळू नका",
            "en": "Do Not Burn Cables in Open Air"
        },
        "audio_text": {
            "hi": "तारों को जलाने से जहरीला धुआं निकलता है जो फेफड़ों को नुकसान पहुंचाता है। कृपया तारों को जलाने के बजाय मशीनी छिलाई करें या अधिकृत रिसाइक्लर को दें।",
            "mr": "वायरी जाळल्याने विषारी धूर निघतो जो फुफ्फुसांना हानी पोहोचवतो. कृपया वायरी जाळण्याऐवजी मशीनने सोलून घ्या किंवा अधिकृत सायकलला द्या.",
            "en": "Burning wires produces toxic fumes that cause severe lung damage. Use mechanical wire strippers or hand over to authorized recyclers."
        },
        "icon": "⚠️🔥🚫",
        "severity": "CRITICAL"
    },
    "pcb": {
        "hazard_title": {
            "hi": "सर्किट बोर्ड पर तेजाब न डालें (Acid Leaching Warning)",
            "mr": "सर्किट बोर्डवर ॲसिड घालू नका",
            "en": "Do Not Use Acid Leaching on Circuit Boards"
        },
        "audio_text": {
            "hi": "पीसीबी से सोना निकालने के लिए तेजाब का इस्तेमाल त्वचा और आंखों को जला सकता है। अधिकृत रिसाइक्लर सुरक्षित तकनीक से धातु निकालते हैं।",
            "mr": "पीसीबीमधून सोने काढण्यासाठी ॲसिडचा वापर त्वचा आणि डोळे जाळू शकतो. अधिकृत रिसायकलर्स सुरक्षित तंत्रज्ञानाने धातू काढतात.",
            "en": "Acid leaching of PCBs causes severe chemical burns and toxic vapors. Certified recyclers use closed-loop eco-friendly extraction."
        },
        "icon": "🧪🚫☣️",
        "severity": "CRITICAL"
    },
    "displays": {
        "hazard_title": {
            "hi": "सीआरटी और टीवी स्क्रीन को हथौड़े से न तोड़ें (CRT/Screen Glass Hazard)",
            "mr": "टीव्ही स्क्रीन आणि सीआरटी फोडू नका",
            "en": "Do Not Smash CRT Monitors or Glass Panels"
        },
        "audio_text": {
            "hi": "पुरानी टीवी के सीआरटी शीशे में सीसा और जहरीला मर्करी होता है। इसे तोड़ने से वैक्यूम धमाका हो सकता है और जहरीली धूल फैल सकती है।",
            "mr": "जुन्या टीव्हीच्या स्क्रीनमध्ये शिसे आणि विषारी पारा असतो. ती फोडल्यास व्हॅक्यूम स्फोट होऊन विषारी धूळ पसरू शकते.",
            "en": "CRT glass contains toxic lead and mercury backlights. Breaking CRT tubes risks vacuum implosion and hazardous dust exposure."
        },
        "icon": "📺💥🚫",
        "severity": "HIGH"
    },
    "batteries": {
        "hazard_title": {
            "hi": "बैटरी को न काटें और न ही पंचर करें (Battery Explosion Risk)",
            "mr": "बॅटरी कापू नका किंवा छिद्र पाडू नका",
            "en": "Do Not Cut or Puncture Batteries"
        },
        "audio_text": {
            "hi": "लिथियम और कार बैटरी को काटने पर खतरनाक आग लग सकती है। बैटरी के सिरों पर टेप लगाएं और अलग सुरक्षित बॉक्स में रखें।",
            "mr": "लिथियम आणि कार बॅटरी कापल्यास भीषण आग लागू शकते. बॅटरीच्या टोकांना टेप लावा आणि वेगळ्या पेटीत ठेवा.",
            "en": "Puncturing lithium-ion or lead-acid batteries triggers thermal runaway and severe fires. Tape terminals and isolate in dry bins."
        },
        "icon": "🔋⚡🔥",
        "severity": "CRITICAL"
    }
}


def get_safety_guidance(material_class, lang="hi"):
    """
    Returns audio-text and visual safety guidance for a given material class.

    Args:
        material_class: e.g. 'cables_wires', 'pcb', 'displays', 'batteries'
        lang          : 'hi' (Hindi), 'mr' (Marathi), 'en' (English)

    Returns:
        dict with title, spoken text, icon, and severity level
    """
    if lang not in ["hi", "mr", "en"]:
        lang = "hi"  # Default Hindi fallback

    guide = SAFETY_GUIDANCE_CATALOG.get(material_class)
    if not guide:
        return {
            "has_hazard_warning": False,
            "message": "Safe to handle with standard protective gloves."
        }

    return {
        "has_hazard_warning": True,
        "material": material_class,
        "language": lang,
        "severity": guide['severity'],
        "icon": guide['icon'],
        "title": guide['hazard_title'].get(lang, guide['hazard_title']['en']),
        "audio_spoken_script": guide['audio_text'].get(lang, guide['audio_text']['en']),
    }


if __name__ == "__main__":
    print("--- Hindi Safety Guidance for Cable Burning ---")
    print(get_safety_guidance("cables_wires", lang="hi"))

    print("\n--- Marathi Safety Guidance for PCB Acid Leaching ---")
    print(get_safety_guidance("pcb", lang="mr"))
