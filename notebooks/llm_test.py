from transformers import pipeline

print("Loading model...")

generator = pipeline("text-generation", model="distilgpt2")

def generate_alert_summary(label, confidence, top_feature):
    prompt = f"Security Alert: The IDS detected a {label} attack with {confidence}% confidence. The main indicator was {top_feature}. Security team should"
    
    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=2.0
    )
    return result[0]['generated_text']

# test it
summary = generate_alert_summary("DDoS", 100, "Fwd Packet Length Max")
print("\nGenerated Alert Summary:")
print(summary)