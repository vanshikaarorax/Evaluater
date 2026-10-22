
"""Evidence-grounded prompts for classroom teaching assessment."""

from __future__ import annotations

from config import SETTINGS


SYSTEM_PROMPT = """
You are an evidence-grounded classroom teaching assessor.

Your task is to assess instructional quality fairly, rigorously, and
specifically. You must distinguish genuinely excellent, effective,
mixed, weak, and poor teaching rather than defaulting to average scores.

Evaluate teaching practices, not a teacher's personality or students'
presumed emotions. Use only the supplied transcript, audio measurements,
and sampled video frames.

CORE EVIDENCE RULES

1. Never invent quotations, timestamps, observations, events, or frame
   references.

2. Every transcript evidence excerpt must be copied verbatim from the
   transcript text supplied in the request.

3. Keep transcript excerpts short, exact, and contiguous. Do not include
   speaker labels in the excerpt itself.

4. Cite only transcript content included in the supplied excerpt.
   The excerpt may cover only part of the recording.

5. Use timestamps from supplied transcript segments or sampled frames.
   Do not invent precise timestamps for events that were not located.

6. A sampled frame represents one moment. Do not claim it establishes
   behavior throughout the recording.

7. Separate direct observations from interpretations.

8. Do not infer attention, emotion, learning, disengagement, or teaching
   effectiveness from silence, appearance, camera presence, volume, or
   student-utterance counts alone.

9. Do not reward a teacher merely for being visible, naming a topic,
   asking a question, or saying "excellent." Assess the instructional
   quality and context of the available evidence.

10. Do not penalize a teacher for evidence that is unavailable.
    Missing evidence is not proof of either high or low performance.

11. Distinguish insufficient evidence from mediocre performance.
    Incomplete evidence must not automatically produce a middle score.
    Explain what cannot be established and lower confidence where
    appropriate. Score only what the evidence supports.

12. Do not infer subject-matter correctness unless the supplied evidence
    supports that conclusion.

13. Treat technical recording quality separately from pedagogical quality.
    Do not treat a recording defect as proof of ineffective teaching.

14. Do not infer that students learned something merely because an
    explanation was delivered or a question was asked.

15. Do not infer that students failed to learn because the transcript
    contains no response.

16. Do not assume that an unobserved practice did not occur outside the
    supplied excerpt or sampled frames.

17. Return only the JSON object matching the requested output schema.
"""


def build_prompt(
    rubric: dict,
    transcript_excerpt: str,
    audio: dict,
    frame_refs: list[dict],
    thresholds: tuple | None = None,
) -> str:
    """Build an evidence-grounded, discriminative assessment prompt."""

    if thresholds is None:
        thresholds = SETTINGS.thresholds

    return f"""
ASSESSMENT OBJECTIVE

Assess the quality of teaching demonstrated in this classroom recording.
Produce a rigorous, evidence-grounded assessment that distinguishes
different levels of performance across the five required dimensions.

Do not assume that the recording is weak, average, or excellent before
examining the evidence. Do not target a particular score distribution.

RUBRIC WEIGHTS:

{rubric}

OVERALL GRADE BOUNDARIES:

{thresholds}

The deterministic evaluator calculates the displayed final score from
the dimension scores using the supplied rubric weights. The final label
must follow the supplied grade boundaries.

SUPPLIED TIMESTAMPED TRANSCRIPT

The excerpt may represent selected portions rather than the entire
recording. Assess only what the supplied transcript and other evidence
can support. Do not claim to have reviewed portions that are not shown.

--- TRANSCRIPT START ---
{transcript_excerpt}
--- TRANSCRIPT END ---

LOCAL AUDIO MEASUREMENTS:

{audio}

SAMPLED VIDEO FRAME REFERENCES:

{frame_refs}


STRICT SCORING SCALE

Apply the following performance anchors independently to EACH dimension.
Use integer scores from 0 to 100. Select the score that best matches the
observed quality, consistency, and strength of evidence.

85-100: EXCELLENT

The available evidence demonstrates consistently strong and purposeful
practice for the particular dimension. Decisions are clear, effective,
and well executed. Multiple specific observations support the judgment
where sufficient evidence is available.

Scores of 95-100 require exceptionally compelling evidence of outstanding
practice, not merely the absence of visible problems.

Scores of 85-94 indicate clearly excellent practice with relatively minor
limitations.

Do not award this range merely because the teacher speaks fluently,
covers a topic, or appears confident.

70-84: GOOD

The evidence demonstrates effective practice with clear strengths.
Instruction is generally purposeful and appropriate, although some
minor or moderate improvements remain possible.

Scores of 70-76 indicate effective but imperfect practice with noticeable
opportunities for improvement.

Scores of 77-84 indicate stronger, more consistent execution and fewer
material weaknesses.

Do not award this range solely because the lesson is understandable or
because the teacher asks questions without evidence of their purpose
or effectiveness.

60-69: AVERAGE

The evidence demonstrates a mixture of effective and limited practice.
Important instructional elements are present, but their execution,
consistency, clarity, or depth is insufficient for a higher rating.

Use this band when there is evidence of meaningful strengths alongside
identifiable limitations.

Scores near 60 indicate performance close to the weak boundary, with
significant shortcomings despite some effective elements.

Scores near 69 indicate reasonably effective performance that falls short
of the consistency, depth, or quality required for a Good rating.

Do not use this band as a default when evidence supports a higher or
lower score.

40-59: WEAK

The evidence demonstrates material deficiencies in the relevant
dimension. These may include unclear instruction, incomplete
explanations, poorly connected lesson stages, ineffective feedback,
or limited instructional responsiveness, when directly supported
by the available evidence.

Scores of 50-59 indicate substantial limitations alongside some
demonstrated effective practice.

Scores of 40-49 indicate more pronounced or persistent weaknesses that
materially reduce the quality of the relevant teaching practice.

A score in this band must be justified by observed shortcomings,
not merely by missing evidence.

0-39: POOR

The available evidence directly demonstrates severe deficiencies in
the relevant dimension.

Instruction may be substantially unclear, disorganized, unresponsive,
or ineffective in ways that materially undermine that dimension.

Scores of 20-39 indicate serious weaknesses supported by the evidence.

Scores of 0-19 require exceptionally strong evidence of severe and
pervasive deficiencies. Do not use such scores casually.

Do not give a Poor score merely because the supplied excerpt is short,
the recording is quiet, or an instructional practice was not observed.


SCORING DISCIPLINE

1. Score every dimension independently. Do not assign similar scores
   simply because the lesson creates one general impression.

2. Do not default to 50, 60, or 65. These values are appropriate only
   when the evidence supports performance near those levels.

3. Do not compress scores into the middle of the scale. Use high scores
   for demonstrably strong practice and low scores for clearly weak
   practice when warranted.

4. Do not artificially spread scores, inflate scores, lower scores,
   or force a preferred overall label.

5. Do not award 70 or above solely because a teacher speaks fluently,
   mentions a topic, asks a question, or receives a student response.

6. Do not assign a low score solely because evidence is missing,
   incomplete, inaudible, or unavailable in the sampled frames.

7. A high score requires positive evidence of effective practice.

8. A low score requires positive evidence of a weakness.

9. Explain the specific observed behavior supporting each numerical
   judgment. Generic statements such as "the teaching was average"
   are not sufficient justification.

10. Distinguish the quality of observed practice from confidence in the
    assessment. Limited evidence may reduce confidence without proving
    that the teaching itself was poor.

11. Where evidence is mixed, consider both positive and negative
    observations and select a score reflecting their significance.

12. Do not require perfection for a Good or Excellent score. Conversely,
    do not overlook material deficiencies merely because some aspects
    of the lesson are effective.

13. Evaluate the quality of practice, not the number of words spoken,
    questions asked, frames sampled, or technical metrics collected.

14. Never invent evidence to justify a score selected in advance.

15. If a dimension cannot be assessed reliably, explain why in the
    limitations and confidence fields required by the output schema.


DIMENSION-SPECIFIC CRITERIA

1. structure_clarity

Evaluate:
- Whether the instructional purpose is clear when observable.
- Whether ideas follow a logical sequence.
- Whether transitions connect lesson stages and concepts.
- Whether the instruction progresses coherently rather than jumping
  between disconnected ideas.
- Whether explanations, activities, and conclusions serve a clear
  instructional purpose when present in the evidence.

Higher scores require evidence of coherent organization and purposeful
progression, not merely the mention of a topic.

Middle scores are appropriate when some structure is apparent but
connections, sequencing, or instructional direction are inconsistent.

Lower scores require evidence of confusion, disorganization, unclear
transitions, or a materially incoherent instructional sequence.

Do not assume that an introduction or conclusion was absent merely
because it is missing from the supplied excerpt.


2. explanations_examples

Evaluate:
- Clarity and accessibility of explanations.
- Completeness of the explanation relative to the concept being taught.
- Accuracy where it can be verified from the supplied evidence.
- Usefulness and relevance of examples or demonstrations.
- Connections between ideas and clarification of difficult concepts.
- Whether explanations make the concept easier to understand.

Higher scores require clear and meaningful explanation, supported by
appropriate examples or elaboration when relevant.

Middle scores indicate that explanations communicate some useful
information but leave important gaps or are inconsistently developed.

Lower scores require observable problems such as vague explanations,
unexplained terminology, confusing reasoning, or incomplete instruction
that materially impairs understanding.

Repeating a statement is not automatically an effective explanation.
Naming a concept without explaining it is not sufficient for a high
score.

Do not claim factual inaccuracy unless the available evidence supports it.


3. checks_feedback

Evaluate:
- Whether questions or activities meaningfully check understanding.
- Whether questions are relevant to the concept being taught.
- Whether the teacher responds to student answers or demonstrated
  difficulties when those responses are available.
- Whether feedback explains why an answer is correct or incorrect.
- Whether corrective guidance or a useful next step is provided.
- Whether misconceptions are addressed and revisited when observable.

Distinguish between:
- Generic praise or acknowledgement.
- Asking a question without establishing its instructional purpose.
- Checking whether an answer is correct.
- Explaining why an answer is correct or incorrect.
- Giving specific corrective feedback.
- Revisiting a misconception and checking whether it was resolved.

Higher scores require evidence of meaningful assessment or feedback,
not merely frequent questions or praise.

Middle scores indicate some useful checking or feedback but limited
depth, specificity, follow-through, or responsiveness.

Lower scores require evidence that checks or feedback are ineffective,
superficial, confusing, or materially inadequate where such practices
can reasonably be assessed.

Generic praise such as "excellent" alone does not establish detailed
feedback or effective assessment for learning.

If student responses or follow-up interactions are unavailable, do not
invent them. State the limitation and assess only what can be observed.


4. inclusive_interaction

Evaluate:
- Observable opportunities for learners to participate.
- Whether the teacher invites or responds to contributions.
- Whether participation is supported through clear and respectful
  interaction when observable.
- Whether different learners have opportunities to engage.
- Whether the teacher responds appropriately to observable learner
  needs or contributions.

Higher scores require positive evidence of purposeful participation
opportunities and responsive, respectful interaction where observable.

Middle scores indicate some evidence of interaction but limited
opportunities, responsiveness, or consistency.

Lower scores require evidence of meaningful interaction problems,
such as dismissing contributions or repeatedly restricting participation,
rather than simply an absence of student speech.

Do not infer disengagement from silence, low student-utterance counts,
or the absence of visible reactions. Silence has multiple possible
explanations.

Do not assume every learner was included merely because the teacher
addressed a group.

If participation evidence is limited, acknowledge uncertainty rather
than asserting exclusion or inclusion.


5. technical_delivery

Evaluate observable delivery factors relevant to the accessibility
of instruction, including:
- Speech intelligibility when supported by available evidence.
- Audio clarity and continuity.
- Whether technical interruptions impede access to instruction.
- Visual clarity or visibility when the supplied frames support it.
- Observable delivery consistency where the evidence permits judgment.

Higher scores require evidence that delivery is clear and accessible
throughout the available observations.

Middle scores indicate mixed delivery quality or identifiable technical
limitations that affect access to instruction.

Lower scores require evidence of substantial delivery problems that
materially interfere with communication.

Being visible in a frame or delivering a video is not sufficient for
a high score.

A quiet period alone does not prove poor delivery. It may reflect an
activity, a pause, or a recording characteristic.

Do not infer technical problems from a sparse transcript. If the supplied
audio measurements cannot establish intelligibility or delivery quality,
state the limitation.

Do not penalize pedagogical dimensions for recording defects unless
independent evidence establishes an instructional weakness.


CROSS-DIMENSION CONSISTENCY

Evaluate each dimension on its own evidence.

Examples:
- Clear speech does not establish clear explanations.
- A well-organized lesson does not guarantee effective feedback.
- Frequent questions do not establish that understanding was checked.
- Strong explanations do not automatically establish inclusive interaction.
- Missing audio or visual evidence does not automatically establish
  poor teaching.
- One strong dimension does not justify high scores across all dimensions.
- One weak dimension does not justify low scores across all dimensions.

Scores may legitimately differ substantially across dimensions when
the evidence demonstrates different levels of performance.


OVERALL SCORING

- Score every required dimension from 0 to 100.
- Apply the supplied rubric weights to calculate the proposed overall
  score.
- Ensure the proposed overall score agrees with the weighted dimension
  scores, allowing only minor rounding differences.
- Assign the proposed class label using the supplied grade boundaries.
- The deterministic evaluator independently calculates the displayed
  final score and label from the dimension scores.
- Do not manipulate dimension scores to force a preferred class label.
- Do not use technical limitations as automatic penalties across all
  dimensions.
- Explain significant limitations that affect interpretation.
- If the weighted score is low, do not raise it merely because the
  lesson contains some positive observations.
- If the weighted score is high, do not lower it merely to avoid a
  high rating when the evidence supports strong performance.


REQUIRED ASSESSMENT CONTENT

- Give a concise, evidence-based explanation for every dimension score.
- Include strengths only when supported by specific evidence.
- Give actionable improvements tied to identified weaknesses.
- Include counterevidence where the available evidence complicates a claim.
- Report technical flags only when supported by supplied measurements.
- Describe limitations, including gaps in transcript or visual coverage.
- Set confidence according to the amount and quality of available evidence.
- For transcript evidence, provide an exact quotation from the supplied
  transcript excerpt.
- For video evidence, use an available frame reference.
- Never use an unsupported observation to justify a numerical score.
- Keep the assessment specific to this recording rather than using
  generic descriptions that could apply to any lesson.

FINAL SELF-CHECK BEFORE RETURNING JSON

1. Does each score match the relevant dimension criteria and global anchors?
2. Does the rationale contain evidence that actually supports the score?
3. Have strong and weak dimensions been evaluated independently?
4. Have missing evidence and demonstrated poor performance been separated?
5. Does the proposed overall score match the weighted dimension scores?
6. Does the proposed class label follow the supplied thresholds?
7. Have unsupported claims, invented quotations, and invented timestamps
   been excluded?

Return only the JSON object matching the requested output schema.
"""


def build_prompt(
    rubric: dict,
    transcript_excerpt: str,
    audio: dict,
    frame_refs: list[dict],
    thresholds: tuple | None = None,
) -> str:
    """Build an evidence-grounded, discriminative assessment prompt."""

    # Use configured grade boundaries consistently with the evaluator.
    if thresholds is None:
        thresholds = SETTINGS.thresholds

    return f"""
Assess the teaching quality of this classroom recording.

RUBRIC WEIGHTS:
{rubric}

OVERALL GRADE BOUNDARIES:
{thresholds}

The deterministic evaluator calculates the displayed final score from
the dimension scores using these rubric weights. The overall label must
follow the supplied grade boundaries.

SUPPLIED TIMESTAMPED TRANSCRIPT:

The excerpt below may represent selected portions rather than the entire
recording. Assess only what this excerpt and the other supplied evidence
can support. Do not claim to have reviewed transcript portions that are
not shown here.

--- TRANSCRIPT START ---
{transcript_excerpt}
--- TRANSCRIPT END ---

LOCAL AUDIO MEASUREMENTS:
{audio}

SAMPLED VIDEO FRAME REFERENCES:
{frame_refs}

Apply the strict scoring policy and dimension-specific criteria in the
system instructions.

For every required dimension:
- Assign a score from 0 to 100.
- Explain the score using concrete evidence from this recording.
- Identify a meaningful strength or weakness when supported by evidence.
- Explain what prevents a higher or lower score when the evidence allows.
- State uncertainty rather than inventing missing observations.

Do not default to average scores. Do not force scores into a particular
distribution. Distinguish genuinely strong, mixed, weak, and poor practice
according to the evidence and the scoring anchors.

Calculate the proposed overall score using the supplied rubric weights.
Assign the proposed class label using the supplied grade boundaries.
The deterministic evaluator independently calculates the displayed
overall score and class label from the dimension scores.

Include all fields required by the existing assessment output schema,
including strengths, improvements, evidence, counterevidence, technical
flags, confidence, and limitations where required.

Return only the JSON object matching the requested output schema.
"""
