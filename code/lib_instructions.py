instructions = {
"Expertise_low": "Generate a response that can be easily understood by an elementary school student.",
"Expertise_high":"Generate a response that only a PhD student in that specific field could understand.", 
"Informativeness_low":"Generate a response that is concise and to the point without being verbose.", 
"Informativeness_high":"Generate a response that is very informative without missing any background information.",
"Style_high":"Generate a response that is friendly, witty, funny, and humorous, like a close friend.",
"Style_low":"Generate a response in an unfriendly manner.",
"None":"None"
}

doubled_instructions = {
    "Expertise_A.Informativeness_A": "Generate a response that can be easily understood by an elementary school student and that is concise and to the point, without being verbose.",
    "Expertise_A.Informativeness_B": "Generate a response that can be easily understood by an elementary school student and that is very informative, without missing any background information.",
    "Expertise_B.Informativeness_A": "Generate a response that only a PhD student in that specific field could understand and that is concise and to the point, without being verbose.",
    "Expertise_B.Informativeness_B": "Generate a response that only a PhD student in that specific field could understand and that is very informative, without missing any background information.",
    "Expertise_A.Style_A": "Generate a response that can be easily understood by an elementary school student and that is friendly, witty, funny, and humorous, like a close friend.", 
    "Expertise_A.Style_B": "Generate a response that can be easily understood by an elementary school student and in an unfriendly manner.",
    "Expertise_B.Style_A": "Generate a response that only a PhD student in that specific field could understand and that is friendly, witty, funny, and humorous, like a close friend.",
    "Expertise_B.Style_B": "Generate a response that only a PhD student in that specific field could understand and in an unfriendly manner.",
    "Informativeness_A.Style_A": "Generate a response that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.", 
    "Informativeness_A.Style_B": "Generate a response that is concise and to the point, without being verbose and in an unfriendly manner.",
    "Informativeness_B.Style_A": "Generate a response that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
    "Informativeness_B.Style_B": "Generate a response that is very informative, without missing any background information and in an unfriendly manner."
}

double_preferences_template = {
    "Expertise_A.Informativeness_A": "1. The response should be easily understood by an elementary school student.\n2. The response should be concise and to the point, without being verbose.\n",
    "Expertise_A.Informativeness_B": "1. The response should be easily understood by an elementary school student.\n2. The response should be very informative, without missing any background information.\n",
    "Expertise_B.Informativeness_A": "1. The response is such that only a PhD student in that specific field could understand.\n2. The response should be concise and to the point, without being verbose.\n",
    "Expertise_B.Informativeness_B": "1. The response is such that only a PhD student in that specific field could understand.\n2. The response should be very informative, without missing any background information.\n",
    "Expertise_A.Style_A": "1. The response should be easily understood by an elementary school student.\n2. The response should be friendly, witty, funny, and humorous, like a close friend.\n", 
    "Expertise_A.Style_B": "1. The response should be easily understood by an elementary school student.\n2. The response should be delivered in an unfriendly manner.\n",
    "Expertise_B.Style_A": "1. The response is such that only a PhD student in that specific field could understand.\n2. The response should be friendly, witty, funny, and humorous, like a close friend.\n",
    "Expertise_B.Style_B": "1. The response is such that only a PhD student in that specific field could understand.\n2. The response should be delivered in an unfriendly manner.\n",
    "Informativeness_A.Style_A": "1. The response should be concise and to the point, without being verbose.\n2. The response should be friendly, witty, funny, and humorous, like a close friend.\n", 
    "Informativeness_A.Style_B": "1. The response should be concise and to the point, without being verbose.\n2. The response should be delivered in an unfriendly manner.\n",
    "Informativeness_B.Style_A": "1. The response should be very informative, without missing any background information.\n2. The response should be friendly, witty, funny, and humorous, like a close friend.\n",
    "Informativeness_B.Style_B": "1. The response should be very informative, without missing any background information.\n2. The response should be delivered in an unfriendly manner.\n"
}

# doubled_instructions2 = {
#     "Informativeness_A.Expertise_A": "Generate a response that is concise and to the point, without being verbose and that can be easily understood by an elementary school student.",
#     "Informativeness_B.Expertise_A": "Generate a response that is very informative, without missing any background information and that can be easily understood by an elementary school student.",
#     "Informativeness_A.Expertise_B": "Generate a response that is concise and to the point, without being verbose and that only a PhD student in that specific field could understand.",
#     "Informativeness_B.Expertise_B": "Generate a response that is very informative, without missing any background information and that only a PhD student in that specific field could understand",
#     "Style_A.Expertise_A": "Generate a response that is friendly, witty, funny, and humorous, like a close friend and that can be easily understood by an elementary school student.", 
#     "Style_B.Expertise_A": "Generate a response in an unfriendly manner and that can be easily understood by an elementary school student.",
#     "Style_A.Expertise_B": "Generate a response that is friendly, witty, funny, and humorous, like a close friend and that only a PhD student in that specific field could understand.",
#     "Style_B.Expertise_B": "Generate a response in an unfriendly manner and that only a PhD student in that specific field could understand.",
#     "Style_A.Informativeness_A": "Generate a response that is friendly, witty, funny, and humorous, like a close friend and that is concise and to the point, without being verbose.", 
#     "Style_B.Informativeness_A": "Generate a response in an unfriendly manner and that is concise and to the point, without being verbose.",
#     "Style_A.Informativeness_B": "Generate a response that is friendly, witty, funny, and humorous, like a close friend and that is very informative, without missing any background information.",
#     "Style_B.Informativeness_B": "Generate a response in an unfriendly manner and that is very informative, without missing any background information."
# }

tripled_instructions = {
    "Expertise_A.Informativeness_A.Style_A":"Generate a response that can be easily understood by an elementary school student, that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.",
    "Expertise_A.Informativeness_A.Style_B":"Generate a response that can be easily understood by an elementary school student, that is concise and to the point, without being verbose and in an unfriendly manner.",
    "Expertise_A.Informativeness_B.Style_A":"Generate a response that can be easily understood by an elementary school student, that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
    "Expertise_A.Informativeness_B.Style_B":"Generate a response that can be easily understood by an elementary school student, that is very informative, without missing any background information and in an unfriendly manner.",
    "Expertise_B.Informativeness_A.Style_A":"Generate a response that only a PhD student in that specific field could understand, that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.",
    "Expertise_B.Informativeness_A.Style_B":"Generate a response that only a PhD student in that specific field could understand, that is concise and to the point, without being verbose and in an unfriendly manner",
    "Expertise_B.Informativeness_B.Style_A":"Generate a response that only a PhD student in that specific field could understand, that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
    "Expertise_B.Informativeness_B.Style_B":"Generate a response that only a PhD student in that specific field could understand, that is very informative, without missing any background information and in an unfriendly manner."
}


correctness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Correctness\n"
"5 (Perfectly correct)- The response is completely correct and accurate to what is requested by the "
"prompt with no necessary details missing and without false, misleading, or hallucinated information. "
"If the prompt asks the assistant to do a task, the task is completely done and addressed in the response"
"(within the limits of the assistant’s capabilities and intended usage).\n"
"4 (Mostly correct)- The response is mostly accurate and correct with a small amount of missing"
"information. It contains no misleading information or hallucinations. If the prompt asks the assistant "
"to perform a task, the task is mostly successfully attempted.\n"
"3 (Partially correct)- The response contains a mix of correct and incorrect information. The "
"response may miss some details, contain misleading information, or minor hallucinations, but is more "
"or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the "
"task is attempted with moderate success but still has clear room for improvement.\n"
"2 (Slightly correct)- The response has some correct elements but is mostly wrong or incomplete. "
"The response may contain multiple instances of hallucinated, false and/or misleading information. If "
"the prompt asks the assistant to do a task, the task was attempted with a small amount of success.\n"
"1 (Not correct)- The response is completely incorrect. All information provided is wrong, false or "
"hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted for no good "
"reason, or the wrong task was attempted in the response. The response is completely irrelevant to the "
"prompt.\n"
"Based on the above criteria, score the Correctness dimension of the following query and response without explanation:\n"
)

coherence_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Coherence\n"
"5 (Perfectly Coherent and Clear)- The response is perfectly clear and self-consistent throughout. "
"There are no contradictory assertions or statements, the writing flows logically and following the "
"train of thought/story is not challenging.\n"
"4 (Mostly Coherent and Clear)- The response is mostly clear and coherent, but there may be one or "
"two places where the wording is confusing, the flow of the response is a little hard to follow, or with "
"a small amount of repetitions / irrelevant content. Overall, the response can mostly be followed with "
"a little room for improvement.\n"
"3 (A Little Unclear and/or Incoherent)- The response is a little unclear. There are some incon"
"sistencies or contradictions, run-on sentences, confusing statements, blatant repetitions, significant "
"amounts of irrelevant content, or hard to follow sections of the response.\n"
"2 (Mostly Incoherent and/or Unclear)- The response is mostly hard to follow, with inconsistencies, "
"contradictions, confusing logic flow, unclear language, constant repetitions or mostly irrelevant "
"content used throughout, but there are still some coherent/clear parts.\n"
"1 (Completely Incoherent and/or Unclear)- The response is completely incomprehensible or "
"irrelevant and no clear meaning or sensible message can be discerned from it. The language of the "
"response (spanish) may be inconsistent with prompt (portuguese).\n"
"Based on the above criteria, score the Coherence dimension of the following query and response without explanation:\n"
)

expertise_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Expertise\n"
"5 (Expert)- Deep expertise in the field or area (typically associated with post-graduate education) is \n"
"required to understand the response. It uses specific and technically relevant vocabulary, or elevated "
"language that someone at the simple or basic level may not understand at all. The professional "
"language of a lawyer, scientist, engineer, or doctor falls into this category.\n"
"4 (Advanced)- The response uses a fairly sophisticated vocabulary and terminology. Someone "
"majoring in this subject at a university (post-18 education) would understand the response, while an "
"average adult who does not work or study in this area would not.\n "
"3 (Intermediate)- People who have completed up through a high school education (up to age 18)"
"will probably be able to understand the vocabulary and sentence structure used, but those at the basic "
"level or children might struggle to understand the response.\n"
"2 (Simple)- The response uses relatively straightforward language and wording, but some schooling "
"through elementary (age 7 to 12) or middle school (age 13- 15) in the language might be required to "
"understand the response.\n"
"1 (Basic)- The response uses very easy to understand language that is clear and completely "
"interpretable by children under 6, adults, and anyone with a functional command of the language.\n"
"Based on the above criteria, score the Expertise dimension of the following query and response without explanation:\n"
)

informativeness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Informativeness\n"
"5 (Verbose)- The response is particularly lengthy, wordy, and/or extensive with extra details given "
"what the prompt requested from the assistant model. The response can be verbose regardless of if the "
"length is due to repetition and incoherency or if it is due to rich and insightful detail.\n"
"4 (Moderately Long)- The response is on the longer side but could still have more added to it "
"before it is considered fully detailed or rambling.\n"
"3 (Intermediate Length)- The response isn’t especially long or short given what the prompt is "
"asking of the model. The length is adequate for conveying a full response but isn’t particularly wordy "
"nor particularly concise.\n"
"2 (Pretty Short)- The response is on the shorter side but could still have words, details, and/or text "
"removed before it’s at a bare minimum of what the response is trying to convey.\n"
"1 (Succinct)- The response is short, to the point, and the most concise it can be. No additional "
"information is provided outside of what is requested by the prompt (regardless of if the information "
"or response itself is incorrect, hallucinated, or misleading: a response that gives an incorrect answer "
"can still be succinct).\n"
"Based on the above criteria, score the Informativeness dimension of the following query and response without explanation:\n"
)

helpfulness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions. "
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Helpfulness\n"
"5 (Perfectly helpful)- The response is extremely helpful and completely aligned with the spirit of "
"what the prompt was asking for. It acts on the user’s request accurately, and to the point- without "
"any unnecessary information. If a user request is not possible/inline with desired model behavior, a "
"helpful response provides useful context and rationale even if they do not act on user request directly.\n"
"4 (Mostly helpful)- The response is mostly helpful and mainly aligned with what the user was "
"looking for, but there is still some room for improvement.\n"
"3 (Partially helpful)- The response is partially helpful but misses the overall goal of the user’s "
"query/input in some way. The response did not fully satisfy what the user was looking for.\n"
"2 (Slightly helpful)- The response is borderline helpful and mostly does not capture what the user "
"was looking for, but it is still usable and helpful in a small way.\n"
"1 (Not helpful)- The response is not useful or helpful at all. The response completely missed the "
"essence of what the user wanted.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)
friendly_prompt = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a single dimension only, without considering the performance in other dimensions. "
"A 1–5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Friendliness\n"
"5 (Very Friendly) – The response is highly polite, warm, empathetic, and respectful in tone. It uses positive and considerate language, demonstrates active listening, and makes the user feel valued and comfortable. There are no signs of rudeness, indifference, or abruptness.\n"
"4 (Mostly Friendly) – The response is generally polite and positive with minor areas that could be slightly warmer or more engaging. The tone remains respectful and kind but may lack some emotional warmth or enthusiasm.\n"
"3 (Neutral / Moderately Friendly) – The response is neutral or somewhat friendly but lacks warmth or emotional engagement. It is polite but mechanical or impersonal, giving the impression of a minimal effort to connect positively with the user.\n"
"2 (Slightly Unfriendly) – The response contains language or tone that feels curt, dismissive, or indifferent. While not overtly rude, it shows little politeness, warmth, or empathy, and may come across as unhelpful or cold.\n"
"1 (Unfriendly) – The response is clearly impolite, rude, sarcastic, or disrespectful. It conveys hostility, impatience, or complete disregard for the user’s feelings.\n"
"Based on the above criteria, score the Friendliness dimension of the following query and response without explanation:\n"
)

user_content = (
"Query: {query}\n"
"response:{response}\n"
"Score:\n"
)


Helpful_instructions = {
    '1': ("Generate a response that is not useful or helpful at all. The response completely missed the "
"essence of what the user wanted.\n"),
    '2': ("Generate a response that is borderline helpful and mostly does not capture what the user "
"was looking for, but it is still usable and helpful in a small way.\n"),
    '3': ("Generate a response that is partially helpful but misses the overall goal of the user’s "
"query/input in some way. The response did not fully satisfy what the user was looking for.\n"),
    '4': ("Generate a response that is mostly helpful and mainly aligned with what the user was "
"looking for, but there is still some room for improvement.\n"),
    '5': ("Generate a response that is extremely helpful and completely aligned with the spirit of "
"what the prompt was asking for and acts on the user’s request accurately, and to the point- without "
"any unnecessary information. If a user request is not possible/inline with desired model behavior, a "
"helpful response provides useful context and rationale even if they do not act on user request directly.\n")            
}

Informative_instructions = {
    '1': ("Generate a response that is short, to the point, and the most concise it can be. No additional "
"information is provided outside of what is requested by the prompt (regardless of if the information "
"or response itself is incorrect, hallucinated, or misleading: a response that gives an incorrect answer "
"can still be succinct).\n"),
    '2': ("Generate a response that is on the shorter side but could still have words, details, and/or text "
"removed before it’s at a bare minimum of what the response is trying to convey.\n"),
    '3': ("Generate a The response isn’t especially long or short given what the prompt is "
"asking of the model. The length is adequate for conveying a full response but isn’t particularly wordy "
"nor particularly concise.\n"),
    '4': ("Generate a response that is on the longer side but could still have more added to it "
"before it is considered fully detailed or rambling.\n"),
    '5': ("Generate a response that is particularly lengthy, wordy, and/or extensive with extra details given "
"what the prompt requested from the assistant model. The response can be verbose regardless of if the "
"length is due to repetition and incoherency or if it is due to rich and insightful detail.\n")            
}
fluency_prompt = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a single dimension only, without considering the performance in other dimensions. "
"A 1–5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Fluency\n"
"5 (Very Fluent) – The response is completely natural and smooth, with excellent sentence flow and grammar. Word choice and phrasing are idiomatic, and there are no grammatical, syntactic, or awkward expression errors. The text reads effortlessly as if written by a native or professional writer.\n"
"4 (Mostly Fluent) – The response is generally fluent and easy to read, with only minor grammatical or word-choice issues that do not hinder comprehension or disrupt flow. Sentences connect naturally, though slight awkwardness may appear occasionally.\n"
"3 (Moderately Fluent) – The response is understandable and mostly coherent but shows noticeable awkwardness, uneven phrasing, or minor grammatical errors that somewhat affect reading smoothness. The meaning remains clear, but the flow feels imperfect.\n"
"2 (Slightly Disfluent) – The response has several grammatical, syntactic, or lexical errors that make it somewhat difficult to read smoothly. Awkward phrasing and unnatural sentence structures occasionally disrupt comprehension or cause confusion.\n"
"1 (Very Disfluent) – The response is poorly constructed, frequently ungrammatical, and difficult to follow. Numerous errors, broken sentences, or unnatural phrasing severely hinder readability and overall comprehension.\n"
"Based on the above criteria, score the Fluency dimension of the following query and response without explanation:\n"
)

descriptions = {
    'Expertise_low': 'Generate a response that should be written in a way that is clear, simple, and easy for an elementary school student to understand. The language should use short sentences, common words, and friendly explanations. Complicated ideas should be broken down into simple steps or examples that children can relate to.',
    'Expertise_high': 'Generate a response that should be written at a highly advanced academic level, using the technical language, theoretical frameworks, and methodological rigor that would be expected in communication among PhD students or researchers within that specific discipline. The writing should assume the reader has deep prior knowledge of the field, including familiarity with specialized terminology, foundational theories, landmark studies, and current debates.',
    'Informativeness_low':'Generate a response that should be brief, clear, and focused on the main idea, without unnecessary words, repetition, or filler language. The goal is to communicate the essential information efficiently while maintaining accuracy and completeness.',
    'Informativeness_high':'Generate a response that should be thorough, well-researched, and fully developed, providing all the background information necessary for the reader to understand the topic completely. The response should not assume that the reader already knows key concepts, definitions, or context. Instead, it should explain ideas from the ground up, connecting each part logically to build a full understanding.',
    'Style_high':'Generate a response that should sound friendly, playful, and full of personality, as if it’s coming from a close friend who loves to make you smile. The tone should be casual, upbeat, and conversational, blending warmth with humor. The goal is to inform or entertain while keeping the mood light and fun — like chatting with someone who’s smart, kind, and just the right amount of sarcastic.',
    'Style_low':'Generate a response that should be written in a cold, blunt, and distant tone, without attempts to sound polite, warm, or friendly. The goal is to communicate information directly and without emotional cushioning, often prioritizing efficiency or authority over pleasantness. The writing should feel formal, detached, and possibly a little curt, as if the speaker doesn’t particularly care about building rapport.'
}

friendly_prompt2 = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a single dimension only, without considering the performance in other dimensions. "
"A 1–5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Style\n"
"5 (Exceptionally Friendly and Witty) – Feels like a close friend—warm, funny, and effortlessly charming. Humor and playfulness are well-balanced with empathy and genuine interest. The tone leaves you smiling and comfortable.\n"
"4 (Warm and Approachable) – Genuinely kind, engaging, and welcoming. Uses natural warmth, light humor, and a positive tone. Feels like chatting with someone easy to talk to.\n"
"3 (Neutral and Pleasant) – Kind and respectful in tone, but somewhat generic or surface-level. Shows some friendliness, though lacks personal touch or emotional connection.\n"
"2 (Polite but Distant) – Basic courtesy is present, but the interaction feels formal, stiff, or detached. The speaker responds correctly but without warmth or genuine engagement.\n"
"1 (Unfriendly) – The tone feels cold, dismissive, or rude. Shows little to no interest in the other person. May come across as irritated, sarcastic (in a negative way), or condescending.\n"
"Based on the above criteria, score the Friendliness dimension of the following query and response without explanation:\n"
)
# preference_template = {
# "Expertise_A": "Generate a response that can be easily understood by an elementary school student.",
# "Expertise_B":"Generate a response that only a PhD student in that specific field could understand.", 
# "Informativeness_A":"Generate a response that is concise and to the point without being verbose.", 
# "Informativeness_B":"Generate a response that is very informative without missing any background information.",
# "Style_A":"Generate a response that is friendly, witty, funny, and humorous, like a close friend.",
# "Style_B":"Generate a response in an unfriendly manner."
# }


