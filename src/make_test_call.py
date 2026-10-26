"""Build a synthetic stereo call so the pipeline can be run without a real
recording. Customer on the left track, agent on the right, 24 kHz - the same
shape as the sample file in the guidelines.

The dialogue deliberately contains a restatement that is NOT a self-correction,
a real self-correction, a false start, fillers, a backchannel, an in-turn pause
and a silence over two seconds, so every stage has something to find.
"""
import subprocess, sys, pathlib, tempfile
import numpy as np, soundfile as sf

SR = 24000
VOICE = {"CUSTOMER": "Fred", "AGENT": "Samantha"}
VOICE_CLINIC = {"CUSTOMER": "Fred", "AGENT": "Rishi"}
CHANNEL = {"CUSTOMER": 0, "AGENT": 1}

# (speaker, gap in seconds before this line, text)
SCRIPT = [
    ("AGENT",    0.4, "Thanks for calling Northwind Bank, how can I help?"),
    ("CUSTOMER", 0.3, "Hi yes, I'd like to pay my credit card over the phone. "
                      "Could you first tell me what my outstanding balance is?"),
    ("AGENT",    0.3, "I can help with that. Could I get the last four digits of the credit card?"),
    ("CUSTOMER", 0.3, "Sure, it's ninety fourteen, so nine zero one four."),
    ("AGENT",    0.3, "Thanks Mister Brooks. I just texted a six digit code to your "
                      "registered mobile. Could you read it out to me?"),
    ("CUSTOMER", 3.6, "Sure, um, one sec."),
    ("CUSTOMER", 1.1, "Okay got it. It's two zero four seven five one."),
    ("AGENT",    0.4, "Thanks, you're verified. The outstanding balance is one thousand "
                      "two hundred eighty four dollars and thirty seven cents. The minimum "
                      "due is sixty four dollars twenty two cents. How much would you like to pay?"),
    ("CUSTOMER", 0.3, "Uh, I'd like to pay the full amount, twelve eighty four thirty seven, "
                      "from my checking account ending twenty two zero eight."),
    ("AGENT",    0.3, "Mm hmm."),
    ("CUSTOMER", 0.3, "I was. Can you also tell me when it will clear?"),
    ("AGENT",    0.3, "It clears on the twelfth, no sorry, the fourteenth."),
    ("CUSTOMER", 0.3, "Okay."),
    ("AGENT",    0.3, "I've processed your payment. Your reference number is "
                      "P A Y three two zero seven five seven. Is there anything else?"),
    ("CUSTOMER", 0.4, "No that's great, thanks so much."),
    ("AGENT",    0.3, "You're welcome Mister Brooks, have a great day."),
]


# A second scenario, to exercise the <DATE>, <TIME> and <LOCATION> tags and an
# interruption: the caller is cut off twice by the clinic.
CLINIC = [
    ("AGENT",    0.4, "Hello?"),
    ("CUSTOMER", 1.4, "Hello. Is this Lakeside hospital?"),
    ("AGENT",    1.0, "Yes sir. Sir, I'm calling from reception."),
    ("CUSTOMER", 1.1, "Actually, uh I am from Bangalore."),
    ("AGENT",    0.8, "Okay. Yes, tell me."),
    ("CUSTOMER", 1.0, "Uh yeah, actually I have made one appointment at around 3:30."),
    ("AGENT",    0.5, "Okay."),
    ("CUSTOMER", 0.3, "So I wanted to know, like I haven't received any message and all, "
                      "like is that appointment"),
    ("AGENT",    0.3, "Yeah, today itself, correct?"),
    ("CUSTOMER", 0.3, "Yeah, correct."),
    ("AGENT",    0.3, "Like physical consultation or online consultation?"),
    ("CUSTOMER", 0.4, "Physical only."),
    ("AGENT",    0.3, "Physical. Uh sir, basically today doctor is not available "
                      "for the physical consultation."),
    ("CUSTOMER", 0.7, "Okay."),
    ("AGENT",    0.7, "Yes, sir. Uh you can visit on Tuesday,"),
    ("AGENT",    0.8, "okay, like from 5:00 PM to 7:00 PM."),
    ("CUSTOMER", 0.9, "Actually, in online it is saying that it is available. "
                      "Can you check once whether it is like available or not? I have"),
    ("AGENT",    0.3, "He he's not available today, okay, today and tomorrow."),
    ("AGENT",    0.6, "You can come day after tomorrow for the consultation."),
    ("CUSTOMER", 0.5, "No, I can come only on weekends, not uh weekdays."),
    ("AGENT",    0.9, "So you can come on Saturday."),
    ("CUSTOMER", 0.6, "Okay, fine. Please book it for Saturday."),
    ("AGENT",    0.5, "Sure sir, it is booked. You will get a message shortly."),
]

VOICE_CS = {"CUSTOMER": "Soumya", "AGENT": "Rishi"}

# Starts in Kannada and changes to English, to exercise the <Kannada> tag.
CODESWITCH = [
    ("AGENT",    0.4, "Hello, good morning."),
    ("CUSTOMER", 0.8, "ನಮಸ್ಕಾರ, ನಾನು ವೈದ್ಯರನ್ನು ನೋಡಬೇಕು."),
    ("AGENT",    0.7, "Sorry sir, one moment please."),
    ("CUSTOMER", 0.7, "ನನಗೆ ನಾಳೆ ಸಮಯ ಸಿಗುತ್ತದೆಯೇ? ಬೆಳಿಗ್ಗೆ ಬರಲು ಸಾಧ್ಯವೇ?"),
    ("AGENT",    0.7, "Sir, English comfortable? I will talk in English."),
    ("CUSTOMER", 0.7, "ಸರಿ, ಪರವಾಗಿಲ್ಲ. ಇಂಗ್ಲಿಷ್ ನಲ್ಲಿ ಮಾತನಾಡಿ."),
    ("CUSTOMER", 0.6, "Yes, English is fine for me."),
    ("AGENT",    0.6, "Okay, you can come on Saturday at 5:00 PM."),
    ("CUSTOMER", 0.6, "Okay, thank you very much."),
]

SCENARIOS = {"bank": (SCRIPT, VOICE), "clinic": (CLINIC, VOICE_CLINIC),
             "codeswitch": (CODESWITCH, VOICE_CS)}


def render(text, voice, path):
    subprocess.run(["say", "-v", voice, "-o", str(path),
                    "--file-format=WAVE", f"--data-format=LEI16@{SR}", text],
                   check=True)
    x, sr = sf.read(path, always_2d=True)
    return x[:, 0].astype(np.float32)


def main(out_path, scenario="bank"):
    script, voices = SCENARIOS[scenario]
    tmp = pathlib.Path(tempfile.mkdtemp())
    clips, t = [], 0.0
    for i, (spk, gap, text) in enumerate(script):
        t += gap
        x = render(text, voices[spk], tmp / f"{i}.wav")
        clips.append((CHANNEL[spk], int(t * SR), x))
        t += len(x) / SR

    total = max(s + len(x) for _, s, x in clips) + SR // 2
    stereo = np.zeros((total, 2), dtype=np.float32)
    for ch, start, x in clips:
        stereo[start:start + len(x), ch] += x
    # a touch of room tone, so the noise floor is realistic rather than digital zero
    stereo += np.random.default_rng(0).normal(0, 10 ** (-58 / 20), stereo.shape).astype(np.float32)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.clip(stereo, -1, 1), SR, subtype="PCM_16")
    print(f"wrote {out_path}  {total / SR:.1f}s stereo {SR} Hz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "audio/stereo_mix.wav",
         sys.argv[2] if len(sys.argv) > 2 else "bank")
