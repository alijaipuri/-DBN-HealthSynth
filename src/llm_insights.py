import os
from groq import Groq
import logging

logger = logging.getLogger(__name__)

class LLMInsights:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def analyze_privacy_report(self, privacy_metrics, ml_metrics):
        prompt = f"""You are a medical AI privacy expert. Analyze this synthetic patient data evaluation:

PRIVACY METRICS:
- Overall Privacy Score: {privacy_metrics.get('overall_privacy_score', 0):.1f}/100 (Grade: {privacy_metrics.get('grade', 'N/A')})
- Membership Inference Risk: {privacy_metrics.get('membership_inference', {}).get('risk_level', 'Unknown')}
- Statistical Fidelity: {privacy_metrics.get('statistical_fidelity', {}).get('statistical_fidelity_score', 0):.1f}/100
- Correlation Preservation: {privacy_metrics.get('correlation_preservation', {}).get('correlation_preservation_score', 0):.1f}/100

ML UTILITY (TSTR vs TRTR):
- Overall TSTR Accuracy: {ml_metrics.get('overall_tstr_accuracy', 0)*100:.1f}%
- Overall TRTR Accuracy: {ml_metrics.get('overall_trtr_accuracy', 0)*100:.1f}%
- Utility Preservation: {ml_metrics.get('utility_preservation', 0):.1f}%

Provide:
1. A concise clinical assessment (3-4 sentences)
2. Top 3 privacy strengths
3. Top 3 improvement recommendations
4. HIPAA/GDPR compliance assessment
5. A futuristic recommendation for deploying this in real hospitals

Keep total response under 400 words. Be specific and technical."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"AI analysis unavailable: {str(e)}"

    def generate_patient_narrative(self, patient_record: dict):
        fields = "\n".join([f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in patient_record.items()])
        prompt = f"""You are a clinical AI assistant. Generate a concise, realistic patient health summary for this synthetic record:

{fields}

Write a 3-sentence clinical narrative as if this were a real patient chart note. 
Include risk factors, likely diagnoses to investigate, and recommended screenings. 
Do NOT mention this is synthetic data."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Narrative unavailable: {str(e)}"

    def explain_dbn_architecture(self, layer_sizes):
        prompt = f"""Explain this Deep Belief Network architecture for healthcare data generation to a medical professional (not a data scientist):
Layer sizes: {layer_sizes}
Explain: 1) What a DBN is 2) How it learns patient patterns 3) How it generates new records 4) Privacy guarantees
Keep it under 200 words, use simple language, mention HIPAA."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
