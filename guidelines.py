# guidelines.py

# WHO Guidelines for Noise Pollution
WHO_GUIDELINES = {
    "Daytime (16 hours)": "Average noise level should not exceed 55 dB (decibels).",
    "Nighttime (8 hours)": "Average noise level should not exceed 40 dB.",
    "Short-term exposure": "Sounds above 85 dB can cause hearing damage if exposed for more than 8 hours.",
    "Immediate risk": "Sounds above 120 dB can cause immediate harm to hearing."
}

# Emergency contacts (India example, you can change for your region)
EMERGENCY_CONTACTS = {
    "Police Helpline": "100",
    "Noise Pollution Complaint (CPCB)": "1800-180-1718",
    "Local Municipal Corporation": "Contact your local office",
    "Ambulance": "102 / 108"
}

def get_who_guidelines():
    return WHO_GUIDELINES

def get_emergency_contacts():
    return EMERGENCY_CONTACTS
