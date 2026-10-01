"""Text side of MaizeGuard: symptom descriptors (known classes), open lexicon (naming), token-type and quality probes."""
CLASSES = ["healthy", "gls", "nlb", "rust"]
DESCRIPTORS = {
    "healthy": ["a uniformly green maize leaf with no lesions or spots",
                "healthy corn leaf tissue with intact parallel veins",
                "a clean green corn leaf blade without discoloration",
                "a smooth glossy green maize leaf surface"],
    "gls": ["narrow rectangular gray to tan lesions bounded by the veins of a corn leaf",
            "small matchstick-like lesions running parallel to the veins of a maize leaf",
            "pale brown blocky spots confined between veins on a corn leaf",
            "many parallel straight-edged gray lesions on a maize leaf"],
    "nlb": ["long elliptical cigar-shaped gray-green to tan lesions on a corn leaf",
            "large spindle-shaped necrotic lesions several centimeters long on a maize leaf",
            "tan boat-shaped blight lesions with wavy margins on a corn leaf",
            "dark gray fungal sporulation inside elongated corn leaf lesions"],
    "rust": ["small round to oval cinnamon-brown powdery pustules scattered over a corn leaf",
             "raised reddish-brown rust pustules on both sides of a maize leaf",
             "a corn leaf dotted with brick-red erupting pustules",
             "dense clusters of rust-colored spore pustules on a maize leaf"],
}
CLASSNAME_PROMPT = {"healthy": "a photo of a healthy corn leaf", "gls": "a photo of a corn leaf with gray leaf spot",
                    "nlb": "a photo of a corn leaf with northern leaf blight", "rust": "a photo of a corn leaf with common rust"}
# open lexicon of maize foliar disorders that are NOT among the trained classes (used only for naming abstained cases)
LEXICON = {
    "nls": ["small oval to oblong tan lesions with darker brown borders on a corn leaf",
            "short rectangular chocolate-brown spots with dark margins on a maize leaf",
            "numerous small brown leaf spots with dark edges scattered on a corn leaf"],
    "sclb": ["tan lesions with buff centers and reddish-brown margins elongated between the veins of a corn leaf",
             "parallel-sided tan southern corn leaf blight lesions on a maize leaf"],
    "srust": ["tiny circular orange pustules densely packed on the upper surface of a corn leaf",
              "bright orange southern rust pustules crowded on a maize leaf"],
    "tarspot": ["small raised glossy black spots like droplets of tar on a corn leaf",
                "black stromata scattered across a maize leaf"],
    "eyespot": ["tiny round translucent spots with tan centers and yellow halos on a corn leaf",
                "small eye-shaped spots with brown rings on a maize leaf"],
    "anthracnose": ["oval water-soaked lesions with brown borders and yellow margins on a corn leaf",
                    "irregular tan anthracnose blight with dark specks on a maize leaf"],
    "goss": ["long gray-green water-soaked streaks with dark freckles on a corn leaf",
             "wavy-edged leaf blight with dark water-soaked freckles on a maize leaf"],
    "mosaic": ["light and dark green mosaic streaking on a corn leaf",
               "mottled yellow-green stripes from a viral infection on a maize leaf"],
    "ndef": ["yellowing in a V shape from the tip along the midrib of a corn leaf",
             "pale yellow older maize leaf from nitrogen deficiency"],
}
TOKEN_PROMPTS = {
    "lesion": ["a close-up of a diseased necrotic lesion on a leaf", "a brown dead spot on plant tissue",
               "a fungal pustule on a leaf", "a gray blighted patch of a leaf"],
    "tissue": ["a close-up of healthy green leaf tissue", "green plant leaf surface", "a fresh green leaf blade"],
    "background": ["soil and ground", "a gray tabletop background", "a human hand", "the sky",
                   "a black background", "other plants and weeds in a field"],
}
QUALITY_PROMPTS = ["a blurry photo", "a dark underexposed photo", "a heavily compressed photo with blocky artifacts",
                   "a noisy grainy photo", "an overexposed photo with glare", "a sharp well-lit clear photo"]
TEMPLATES_ZS = ["a photo of {}.", "a close-up photo of {}.", "a field photo of {}.", "an agricultural photo of {}."]
