"""
Unit and Integration Test for 'How DCGAN Works' Interactive Feature.
Verifies:
1. Flask dashboard HTTP 200 response.
2. Complete visual representation flow nodes (Latent path, Real path, Discriminator, Loss, Gradients, Adam, Loop).
3. Exact wording for Steps 1 through 5.
4. Exact wording for Viva Callout.
5. Interactive stepper controls and indicators.
6. API generate endpoint functionality and per-image Discriminator scores.
"""

import json
import urllib.request
import sys


def test_dashboard_server():
    url = "http://127.0.0.1:5000/"
    req = urllib.request.urlopen(url)
    assert req.status == 200, f"Expected 200, got {req.status}"
    html = req.read().decode("utf-8")

    # 1. Section Presence
    assert 'id="how-dcgan-works"' in html, "Missing #how-dcgan-works section"
    assert "How DCGAN Works" in html, "Missing section heading 'How DCGAN Works'"
    print("[PASS] [1/6] Section #how-dcgan-works is present with HTTP 200.")

    # 2. Controls & Stepper Pills
    assert 'id="step-prev-btn"' in html, "Missing #step-prev-btn"
    assert 'id="step-next-btn"' in html, "Missing #step-next-btn"
    assert 'id="step-indicators"' in html, "Missing #step-indicators"
    assert "jumpToStep(0)" in html, "Missing step jump handler"
    assert "jumpToStep(4)" in html, "Missing step 5 jump handler"
    print("[PASS] [2/6] Stepper controls (Next, Prev, Indicators 1-5) verified.")

    # 3. Flow Nodes (Dual paths + downstream pipeline)
    expected_nodes = [
        "node-noise",        # Random Latent Noise
        "node-latent-z",     # 100-dimensional latent vector z
        "node-generator",    # Generator
        "node-gen-image",    # Generated 28x28 Fashion Image
        "node-real-image",   # Fashion-MNIST Real Image
        "node-discriminator",# Discriminator
        "node-prediction",   # Real / Fake prediction
        "node-loss",         # BCE Loss
        "node-gradients",    # Gradients
        "node-weights",      # Adam updates Generator/Discriminator weights
        "node-loop",         # Next batch -> Repeat
    ]
    for node_id in expected_nodes:
        assert f'id="{node_id}"' in html, f"Missing flow node id '{node_id}'"
    print(f"[PASS] [3/6] All {len(expected_nodes)} visual architecture flow nodes verified.")

    # 4. Prompt Exact Step Titles & Explanations
    expected_steps = [
        ("Sample Random Noise", "A random 100-dimensional latent vector z is sampled."),
        ("Generator Creates Image", "The Generator transforms z into a 28×28 synthetic fashion image."),
        ("Discriminator Judges", "The Discriminator receives real and generated images and predicts whether each is real or fake."),
        ("Calculate Loss", "Binary Cross Entropy measures prediction error. Gradients indicate how model parameters should change."),
        ("Update Weights", "Adam updates the Generator and Discriminator weights. New batches then continue the learning process.")
    ]
    for title, explanation in expected_steps:
        assert title in html, f"Missing step title: {title}"
        assert explanation in html, f"Missing step explanation: {explanation}"
    print("[PASS] [4/6] All 5 step titles and exact explanations verified verbatim.")

    # 5. Important Viva Concept Callout
    assert "Input that stays random:" in html, "Missing 'Input that stays random:'"
    assert "Parameters that learn:" in html, "Missing 'Parameters that learn:'"
    assert "Generator/Discriminator weights" in html, "Missing 'Generator/Discriminator weights'"
    print("[PASS] [5/6] Viva Concept Callout matches exact prompt specifications.")

    # 6. Existing Generator API Integration Check
    api_req = urllib.request.Request("http://127.0.0.1:5000/api/generate", method="POST")
    api_res = urllib.request.urlopen(api_req)
    assert api_res.status == 200, f"API generate failed with {api_res.status}"
    data = json.loads(api_res.read().decode("utf-8"))
    assert data.get("status") == "success", f"API status is {data.get('status')}"
    assert len(data.get("predictions", [])) == 16, "Expected 16 per-image predictions"
    print(f"[PASS] [6/6] Existing /api/generate endpoint verified (16 items, {data.get('device')} device).")

    print("\nALL 6 VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dashboard_server()
