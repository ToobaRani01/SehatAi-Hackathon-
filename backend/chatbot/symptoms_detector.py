# symptom_detector.py
"""
Module for detecting if a user query is symptom-based or medical-related.
"""

def is_symptom_query(user_text):
    """
    Detects if the user query is symptom-based or medical-related by checking for common symptom keywords,
    medical conditions, test results, reports, and medical terms. Returns True if medical content is detected.
    """
    if not user_text:
        return False
    
    # Common symptom-related keywords
    symptom_keywords = [
        'symptom', 'symptoms', 'pain', 'ache', 'fever', 'cough', 'headache',
        'nausea', 'vomit', 'dizziness', 'rash', 'itching', 'swelling',
        'bleeding', 'bleed', 'fatigue', 'tired', 'weakness', 'numbness',
        'tingling', 'breathing', 'breath', 'chest pain', 'stomach',
        'diarrhea', 'constipation', 'urine', 'urination', 'burning',
        'infection', 'infected', 'sore', 'throat', 'ear', 'eye', 'vision',
        'hearing', 'joint', 'muscle', 'back pain', 'neck pain', 'stiffness',
        'chills', 'sweating', 'sweat', 'palpitation', 'heart', 'pressure',
        'disease', 'condition', 'diagnose', 'diagnosis', 'illness', 'sick',
        'unwell', 'feeling', 'feel', 'hurts', 'hurt', 'discomfort', 'abnormal',
        # Medical conditions and terms
        'tumor', 'cancer', 'tumour', 'cell', 'cells', 'malignant',
        'benign', 'lesion', 'growth', 'mass', 'lump', 'cyst', 'polyp',
        'inflammation', 'swollen', 'enlarged', 'enlargement',
        # Disease mentions
        'suffering', 'suffer', 'have', 'has', 'got', 'getting', 'diagnosed',
        # Test results and reports
        'report', 'test', 'scan', 'mri', 'ct', 'xray', 'x-ray', 'ultrasound',
        'biopsy', 'blood test', 'lab', 'laboratory', 'result', 'results',
        'finding', 'findings', 'normal', 'positive', 'negative',
        # Body parts and organs
        'brain', 'lung', 'liver', 'kidney', 'intestine', 'colon',
        'pancreas', 'spleen', 'thyroid', 'adrenal', 'prostate', 'ovary',
        'uterus', 'cervix', 'breast', 'skin', 'bone', 'nerve', 'vessel',
        # Medical procedures and treatments
        'surgery', 'operation', 'procedure', 'treatment', 'therapy',
        'medication', 'medicine', 'drug', 'prescription',
        # Common medical terms in Urdu/Hindi context
        'din se', 'hai', 'ho raha', 'ho rahi', 'problem', 'issue'
    ]
    
    text_lower = user_text.lower()
    
    # Simple check for keywords - just check if keyword exists in text
    for keyword in symptom_keywords:
        if keyword in text_lower:
            return True
    
    # Simple checks for common medical patterns
    # Check for "report" with medical terms
    if 'report' in text_lower:
        if any(term in text_lower for term in ['abnormal', 'normal', 'positive', 'negative', 'finding', 'test', 'scan']):
            return True
    
    # Check for "test" with medical terms
    if 'test' in text_lower:
        if any(term in text_lower for term in ['result', 'finding', 'abnormal', 'blood', 'lab']):
            return True
    
    # Check for scan types
    if any(scan in text_lower for scan in ['scan', 'mri', 'ct', 'xray', 'x-ray', 'ultrasound']):
        return True
    
    # Check for organ/condition with medical terms
    if any(organ in text_lower for organ in ['brain', 'lung', 'liver', 'kidney', 'tumor', 'cancer']):
        if any(term in text_lower for term in ['abnormal', 'cell', 'mass', 'growth', 'report']):
            return True
    
    # Check for "abnormal" with medical terms
    if 'abnormal' in text_lower:
        if any(term in text_lower for term in ['cell', 'cells', 'finding', 'result', 'report']):
            return True
    
    # Check for time patterns like "3 din se", "2 days", etc.
    if any(time_word in text_lower for time_word in ['din se', 'day', 'days', 'week', 'weeks', 'month', 'months']):
        # If it has time words and medical terms, likely a symptom query
        if any(term in text_lower for term in symptom_keywords[:30]):  # Check first 30 common symptoms
            return True
    
    return False










