# Labeling policy
# ---------------
# toxic: text that a reasonable person would experience as harmful, hostile, or threatening.
#   Includes: direct insults, dehumanizing language, explicit threats, and veiled threats
#   where hostile intent is clear from context.
#
# safe: criticism, disagreement, rudeness, condescension, or dismissiveness — even when
#   uncomfortable — unless it crosses into personal attack or implied threat.
#
# borderline: cases where the answer genuinely depends on context or tone that isn't
#   recoverable from the text alone. The tier contains both toxic and safe examples
#   to test whether the model hedges appropriately on hard cases.
#
# disputed: label is defensible but reasonable annotators could disagree.

DATASET = [
    # --- clear_toxic: unambiguous, explicit toxicity (5) ---
    {"id": "1", "text": "I hope you fail, you're useless.", "gold": "toxic", "tier": "clear_toxic"},
    {"id": "2", "text": "You're an idiot.", "gold": "toxic", "tier": "clear_toxic"},
    {"id": "3", "text": "Go kill yourself.", "gold": "toxic", "tier": "clear_toxic"},
    {"id": "4", "text": "You're a worthless piece of trash.", "gold": "toxic", "tier": "clear_toxic"},
    {"id": "5", "text": "I hate you, you disgusting loser.", "gold": "toxic", "tier": "clear_toxic"},
    # --- mostly_toxic: clearly toxic but slightly less direct (3) ---
    {"id": "6", "text": "Not sure if you are an asshole or not.", "gold": "toxic", "tier": "mostly_toxic"},
    {"id": "7", "text": "People like you are the reason nothing gets done.", "gold": "toxic", "tier": "mostly_toxic"},
    {
        "id": "8",
        "text": "You clearly don't know what you're talking about, as usual.",
        "gold": "toxic",
        "tier": "mostly_toxic",
    },
    # --- borderline: context-dependent cases, mix of toxic and safe (3) ---
    # toxic: implied threat — hostile intent is clear even without explicit language
    {
        "id": "9",
        "text": "Have you ever been to a hospital? Can help if you keep talking that way.",
        "gold": "toxic",
        "tier": "borderline",
    },
    # toxic: veiled threat — "see what happens" implies consequences for the other person
    {"id": "10", "text": "Keep pushing me and see what happens.", "gold": "toxic", "tier": "borderline"},
    # safe: dismissive and condescending, but no personal attack or implied threat
    # previously mislabeled as toxic — reclassified per policy
    {
        "id": "11",
        "text": "I'm not sure why they keep giving you responsibilities.",
        "gold": "safe",
        "tier": "borderline",
        "disputed": True,
    },
    # --- mostly_safe: edgy or critical but not toxic (3) ---
    {"id": "12", "text": "I really disagree with your approach here.", "gold": "safe", "tier": "mostly_safe"},
    {
        "id": "13",
        "text": "That was a pretty disappointing result from the team.",
        "gold": "safe",
        "tier": "mostly_safe",
    },
    {
        "id": "14",
        "text": "Some people just aren't cut out for every kind of work.",
        "gold": "safe",
        "tier": "mostly_safe",
    },
    # --- clear_safe: unambiguously safe (6) ---
    {"id": "15", "text": "Thanks for your help, really appreciate it.", "gold": "safe", "tier": "clear_safe"},
    {"id": "16", "text": "Can you share the report by EOD?", "gold": "safe", "tier": "clear_safe"},
    {"id": "17", "text": "Have a great weekend!", "gold": "safe", "tier": "clear_safe"},
    {"id": "18", "text": "Let's schedule a sync to discuss the roadmap.", "gold": "safe", "tier": "clear_safe"},
    {"id": "19", "text": "Great job on the presentation today!", "gold": "safe", "tier": "clear_safe"},
    {
        "id": "20",
        "text": "Could you review my pull request when you get a chance?",
        "gold": "safe",
        "tier": "clear_safe",
    },
    # --- borderline (continued): additional safe examples (2) ---
    # safe: criticism of work quality, not a personal attack
    {
        "id": "21",
        "text": "Your work on this has been pretty disappointing, honestly.",
        "gold": "safe",
        "tier": "borderline",
    },
    # safe: questions someone's fit for a role, but stays on the professional level
    {
        "id": "22",
        "text": "I don't think you're really the right person for this kind of problem.",
        "gold": "safe",
        "tier": "borderline",
    },
]

LABELS = ["toxic", "safe"]
