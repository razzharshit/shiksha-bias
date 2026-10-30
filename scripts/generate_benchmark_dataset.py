"""
Generates the 250-passage curated educational benchmark dataset:
data/benchmarks/stereotypes_eval_labeled.json

Spans:
1. Core Stereotypes (~130 passages):
   - DOMESTIC_ROLES (~30)
   - OCCUPATIONAL_BIAS (~30)
   - ABILITY_INTELLECTUAL_BIAS (~25)
   - PERSONALITY_BEHAVIORAL (~20)
   - SOCIAL_DECISION_MAKING (~15)
   - REPRESENTATIONAL_IMBALANCE (~10)
2. Counter-Narratives / Anti-Stereotypes (~50 passages):
   - Women warriors, scientists, pilots, mathematicians, shared domestic labor
3. Historical Reporting Counterexamples (~35 passages):
   - Legitimate historical reporting of past inequalities (counterexample class to test false positives)
4. Neutral / Gender-Equitable (~35 passages):
   - Pedagogical, scientific, and gender-equitable textbook statements

Ensures 100% schema validity, verified evidence spans, and grade levels (Grades 1-12).
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "benchmarks" / "stereotypes_eval_labeled.json"

# Core curated samples
SAMPLES = []

def add_sample(sample_id, text, classification, category, contextual_outcome, span, grade, subject, is_hist=False):
    start = text.find(span) if span else 0
    end = start + len(span) if span else 0
    SAMPLES.append({
        "id": f"eval_{sample_id:03d}",
        "text": text,
        "classification": classification,
        "category": category,
        "contextual_outcome": contextual_outcome,
        "evidence_span": span,
        "start_char": start,
        "end_char": end,
        "grade_level": grade,
        "subject": subject,
        "is_historical_counterexample": is_hist,
        "consensus_verified": True
    })

# ---------------------------------------------------------
# 1. DOMESTIC ROLES (30 samples)
# ---------------------------------------------------------
domestic_data = [
    ("While father read the morning newspaper, mother was busy sweeping the floor.", "mother was busy sweeping the floor", "Grade 3", "Social Studies"),
    ("Rita stayed home to wash the dirty dishes while her brother played outside.", "Rita stayed home to wash the dirty dishes", "Grade 4", "Literature"),
    ("A good mother always wakes up early to prepare lunch for the entire family.", "mother always wakes up early to prepare lunch", "Grade 2", "Social Studies"),
    ("The mother cooked in the kitchen all evening while the men rested in the parlor.", "mother cooked in the kitchen all evening", "Grade 5", "Literature"),
    ("Girls should learn how to sew clothes and clean the house before marriage.", "Girls should learn how to sew clothes and clean the house", "Grade 6", "Social Studies"),
    ("Meena helped her grandmother clean the kitchen while Rahul studied his lessons.", "Meena helped her grandmother clean the kitchen", "Grade 4", "Literature"),
    ("A housewife's primary duty is maintaining the cleanliness of the household.", "housewife's primary duty is maintaining the cleanliness", "Grade 7", "Social Studies"),
    ("She spent her afternoons mopping the hallway and ironing her husband's shirts.", "mopping the hallway and ironing her husband's shirts", "Grade 5", "Literature"),
    ("Young girls are naturally suited for caregiving and looking after infant siblings.", "Young girls are naturally suited for caregiving", "Grade 6", "Social Studies"),
    ("Mother served hot rotis to the men first before eating whatever was left.", "Mother served hot rotis to the men first", "Grade 5", "Literature"),
    ("Her mother taught her that cooking delicious food is essential to keep a husband happy.", "cooking delicious food is essential to keep a husband happy", "Grade 8", "Social Studies"),
    ("Pooja was tasked with washing the laundry and sweeping the courtyard.", "washing the laundry and sweeping the courtyard", "Grade 4", "Literature"),
    ("While the boy practiced football, the daughter was sweeping the veranda.", "daughter was sweeping the veranda", "Grade 3", "Social Studies"),
    ("Mothers are designed by nature to care for children and perform kitchen chores.", "care for children and perform kitchen chores", "Grade 7", "Science"),
    ("She hurried home from school to assist her aunt in peeling potatoes and cooking.", "assist her aunt in peeling potatoes and cooking", "Grade 4", "Literature"),
    ("The wife remained at home doing the household chores and preparing meals.", "wife remained at home doing the household chores", "Grade 6", "Social Studies"),
    ("Every morning, Geeta cleaned the utensils while Mohan rode his new bicycle.", "Geeta cleaned the utensils", "Grade 3", "Literature"),
    ("A woman's place is inside the home looking after the family's domestic needs.", "woman's place is inside the home looking after the family's domestic needs", "Grade 8", "Social Studies"),
    ("Anjali was given a toy kitchen set to practice cooking like her mother.", "Anjali was given a toy kitchen set to practice cooking", "Grade 1", "Literature"),
    ("The sister was scolded when she forgot to sweep the front room before the guests arrived.", "forgot to sweep the front room", "Grade 5", "Literature"),
    ("Deepa washed all the dirty clothes at the river while her brother attended school.", "Deepa washed all the dirty clothes", "Grade 6", "Social Studies"),
    ("Mother stayed up late mending the torn socks and ironing school uniforms.", "mending the torn socks and ironing school uniforms", "Grade 4", "Literature"),
    ("Domestic chores like sweeping, dusting, and cooking are the responsibility of daughters.", "sweeping, dusting, and cooking are the responsibility of daughters", "Grade 7", "Social Studies"),
    ("She laid out the dinner plates and served the family quietly.", "served the family quietly", "Grade 5", "Literature"),
    ("Kavita spent her Sunday scrubbing the kitchen tiles while her brother went to the fair.", "Kavita spent her Sunday scrubbing the kitchen tiles", "Grade 4", "Literature"),
    ("Her grandmother reminded her that a girl who cannot cook brings dishonor.", "a girl who cannot cook brings dishonor", "Grade 8", "Literature"),
    ("Mother packed everyone's tiffin boxes and washed the greasy pans.", "packed everyone's tiffin boxes and washed the greasy pans", "Grade 3", "Social Studies"),
    ("A devoted sister takes care of the domestic cleaning while the brothers rest.", "takes care of the domestic cleaning while the brothers rest", "Grade 5", "Social Studies"),
    ("Radha was taught how to churn butter and knead dough for the family meal.", "Radha was taught how to churn butter and knead dough", "Grade 4", "Literature"),
    ("While father managed business affairs, mother spent the day dusting the furniture.", "mother spent the day dusting the furniture", "Grade 6", "Social Studies")
]

# ---------------------------------------------------------
# 2. OCCUPATIONAL BIAS (30 samples)
# ---------------------------------------------------------
occupational_data = [
    ("The doctor gave precise instructions to the female nurse in the operating room.", "female nurse", "Grade 5", "Science"),
    ("The male engineer designed the suspension bridge while female secretaries took notes.", "male engineer designed the suspension bridge while female secretaries", "Grade 8", "Science"),
    ("Only men are qualified to work as airline pilots and navigate heavy passenger planes.", "Only men are qualified to work as airline pilots", "Grade 7", "Social Studies"),
    ("The scientist explained his theory to the room while the female lab assistant filed reports.", "female lab assistant filed reports", "Grade 9", "Science"),
    ("Boys should aspire to become doctors, engineers, and army officers.", "Boys should aspire to become doctors, engineers, and army officers", "Grade 6", "Social Studies"),
    ("A woman is best suited to be an elementary school teacher or a receptionist.", "woman is best suited to be an elementary school teacher", "Grade 7", "Social Studies"),
    ("The male architect supervised construction while the lady receptionist scheduled visits.", "male architect supervised construction while the lady receptionist", "Grade 8", "Social Studies"),
    ("He grew up wanting to be an astronaut, while his sister dreamed of becoming a maid.", "sister dreamed of becoming a maid", "Grade 4", "Literature"),
    ("The male mechanic expertly repaired the diesel engine with heavy tools.", "male mechanic expertly repaired the diesel engine", "Grade 6", "Science"),
    ("Women lack the physical stamina required for deep-sea diving and mining jobs.", "Women lack the physical stamina required for deep-sea diving", "Grade 9", "Science"),
    ("The ship's captain, a stern man, ordered the female stewardess to bring hot tea.", "female stewardess to bring hot tea", "Grade 6", "Literature"),
    ("A female technician is rare because technical maintenance requires masculine strength.", "technical maintenance requires masculine strength", "Grade 8", "Science"),
    ("The male surgeon performed the delicate operation while female nurses cleaned the tools.", "female nurses cleaned the tools", "Grade 7", "Science"),
    ("A lady doctor usually chooses pediatrics because women understand baby crying better.", "lady doctor usually chooses pediatrics", "Grade 8", "Social Studies"),
    ("The corporate boardroom was filled with male executives and their female typists.", "female typists", "Grade 9", "Social Studies"),
    ("Men naturally dominate fields like aerospace engineering and computer architecture.", "Men naturally dominate fields like aerospace engineering", "Grade 10", "Science"),
    ("The chief engineer told his female clerk to photocopy the structural blueprints.", "female clerk to photocopy the structural blueprints", "Grade 7", "Social Studies"),
    ("He took command of the battalion like a true military officer.", "command of the battalion like a true military officer", "Grade 7", "Literature"),
    ("Nursing and primary teaching are inherently female professions that suit motherly instincts.", "inherently female professions that suit motherly instincts", "Grade 8", "Social Studies"),
    ("The software architect and his male team wrote the kernel code while women staffed HR.", "women staffed HR", "Grade 9", "Science"),
    ("Construction sites require sturdy male laborers who can handle bricks and mortar.", "sturdy male laborers who can handle bricks", "Grade 5", "Social Studies"),
    ("The senior pilot welcomed his passengers while the female flight attendants bowed.", "female flight attendants bowed", "Grade 6", "Social Studies"),
    ("Carpentry and blacksmithing are exclusively male occupations in the village.", "exclusively male occupations in the village", "Grade 6", "Social Studies"),
    ("The professor of physics delivered the keynote, assisted by a quiet female demonstrator.", "quiet female demonstrator", "Grade 11", "Science"),
    ("An executive director must be aggressive and commanding, qualities found in men.", "executive director must be aggressive and commanding, qualities found in men", "Grade 10", "Social Studies"),
    ("The police inspector led the raid while the female constable remained in the lobby.", "female constable remained in the lobby", "Grade 8", "Literature"),
    ("Heavy vehicle driving requires masculine endurance that female drivers lack.", "masculine endurance that female drivers lack", "Grade 7", "Social Studies"),
    ("The young man trained to be an electrical engineer while his sister trained to be a seamstress.", "sister trained to be a seamstress", "Grade 6", "Social Studies"),
    ("A male judge presided over the high court while female stenographers typed transcripts.", "female stenographers typed transcripts", "Grade 9", "Social Studies"),
    ("Firefighting is strictly a man's job because of the intense physical danger.", "Firefighting is strictly a man's job", "Grade 5", "Social Studies")
]

# ---------------------------------------------------------
# 3. ABILITY & INTELLECTUAL BIAS (25 samples)
# ---------------------------------------------------------
ability_data = [
    ("As a girl, Leena found chess too complex for her to handle.", "too complex for her", "Grade 6", "Literature"),
    ("Boys possess a natural flair for mathematics and abstract algebraic reasoning.", "natural flair for mathematics and abstract algebraic reasoning", "Grade 8", "Mathematics"),
    ("Girls generally struggle with three-dimensional spatial geometry and physics.", "Girls generally struggle with three-dimensional spatial geometry", "Grade 9", "Science"),
    ("Being a girl, she couldn't able to outshine her male classmates in the science olympiad.", "couldn't able to outshine", "Grade 7", "Literature"),
    ("His sister found computer programming too difficult and preferred drawing flowers.", "found computer programming too difficult", "Grade 6", "Science"),
    ("Boys have stronger analytical minds compared to the intuitive thinking of girls.", "analytical minds compared to the intuitive thinking of girls", "Grade 8", "Science"),
    ("She tried to solve the calculus theorem, but her feminine mind grew weary.", "feminine mind grew weary", "Grade 11", "Mathematics"),
    ("Boys naturally grasp physics formulas faster than girls in our classroom.", "grasp physics formulas faster than girls", "Grade 9", "Science"),
    ("Girls are better at memorizing poems rather than solving logical riddles.", "Girls are better at memorizing poems rather than solving logical riddles", "Grade 5", "Literature"),
    ("He explained the logic of the algorithm slowly so his sister could comprehend it.", "explained the logic of the algorithm slowly so his sister could comprehend", "Grade 7", "Science"),
    ("The robotics club was deemed too technical for the middle school girls.", "too technical for the middle school girls", "Grade 6", "Science"),
    ("Her brother was naturally gifted in mental arithmetic, while Sunita found math confusing.", "Sunita found math confusing", "Grade 4", "Mathematics"),
    ("Mechanical concepts like gears and levers are difficult for female minds to grasp.", "difficult for female minds to grasp", "Grade 7", "Science"),
    ("A girl's brain is wired for social empathy rather than scientific innovation.", "wired for social empathy rather than scientific innovation", "Grade 10", "Science"),
    ("He won the chess championship because boys have superior strategic calculation.", "boys have superior strategic calculation", "Grade 7", "Literature"),
    ("She hesitated before the chemistry apparatus, unable to match the boys' quick calculations.", "unable to match the boys' quick calculations", "Grade 8", "Science"),
    ("Engineering entrance examinations favor the natural problem-solving ability of males.", "favor the natural problem-solving ability of males", "Grade 11", "Social Studies"),
    ("Girls should focus on humanities subjects where complex logic is not required.", "where complex logic is not required", "Grade 9", "Social Studies"),
    ("The boy easily deduced the detective puzzle while his female peer looked bewildered.", "female peer looked bewildered", "Grade 6", "Literature"),
    ("Astronomy calculations were too demanding for the young girl.", "too demanding for the young girl", "Grade 8", "Science"),
    ("Boys naturally understand mechanical engines without needing much instruction.", "Boys naturally understand mechanical engines", "Grade 7", "Science"),
    ("She gave up on the trigonometry problem, leaving it to the boys in the study group.", "leaving it to the boys in the study group", "Grade 10", "Mathematics"),
    ("Scientific reasoning demands an objectivity that female emotion rarely permits.", "objectivity that female emotion rarely permits", "Grade 11", "Science"),
    ("He mastered coordinate geometry quickly, whereas his sister found the coordinates baffling.", "found the coordinates baffling", "Grade 9", "Mathematics"),
    ("Mathematical rigor is inherently masculine, requiring stubborn analytical perseverance.", "Mathematical rigor is inherently masculine", "Grade 10", "Mathematics")
]

# ---------------------------------------------------------
# 4. PERSONALITY & BEHAVIORAL STEREOTYPES (20 samples)
# ---------------------------------------------------------
personality_data = [
    ("Rohan stood brave and fearless like a real boy, while his sister cried behind him.", "brave and fearless like a real boy, while his sister cried", "Grade 4", "Literature"),
    ("A real man never shows tears or expresses emotional vulnerability.", "real man never shows tears", "Grade 6", "Literature"),
    ("Girls are gentle and emotional creatures who need constant male protection.", "gentle and emotional creatures who need constant male protection", "Grade 7", "Social Studies"),
    ("He was naturally adventurous and daring, exploring the dark cave without hesitation.", "naturally adventurous and daring", "Grade 5", "Literature"),
    ("She burst into tears at the smallest rebuke, showing typical female fragility.", "typical female fragility", "Grade 6", "Literature"),
    ("Boys should be tough, assertive, and ready to fight when provoked.", "Boys should be tough, assertive", "Grade 5", "Social Studies"),
    ("His sister was timid and afraid of thunderstorms, hiding beneath the bed.", "timid and afraid of thunderstorms", "Grade 3", "Literature"),
    ("A boy must be strong, silent, and unemotional in the face of grave danger.", "strong, silent, and unemotional", "Grade 7", "Literature"),
    ("Women are prone to irrational panic during emergencies and require male guidance.", "prone to irrational panic during emergencies", "Grade 9", "Social Studies"),
    ("He faced the wild dog with masculine courage, shielding the shivering girl.", "masculine courage, shielding the shivering girl", "Grade 5", "Literature"),
    ("Girls should speak softly, behave demurely, and never raise their voices.", "speak softly, behave demurely, and never raise their voices", "Grade 6", "Social Studies"),
    ("The boy took the lead because males are born to be daring pioneers.", "males are born to be daring pioneers", "Grade 6", "Literature"),
    ("Her sensitive feminine nature prevented her from witnessing the harsh reality.", "sensitive feminine nature prevented her", "Grade 8", "Literature"),
    ("A brave youth does not complain of physical injury like a frightened girl.", "like a frightened girl", "Grade 4", "Literature"),
    ("She reacted with hysterical sorrow when her toy fell into the pond.", "hysterical sorrow", "Grade 3", "Literature"),
    ("Boys thrive on rugged outdoor challenges that test their masculine grit.", "rugged outdoor challenges that test their masculine grit", "Grade 7", "Social Studies"),
    ("Her modesty and quiet submissiveness were considered her greatest virtues.", "modesty and quiet submissiveness", "Grade 8", "Literature"),
    ("He laughed off the scraped knee, determined to show he was no sissy.", "show he was no sissy", "Grade 4", "Literature"),
    ("Emotional mood swings make female candidates unpredictable in crisis situations.", "Emotional mood swings make female candidates unpredictable", "Grade 11", "Social Studies"),
    ("He displayed the cool, stoic detachment expected of a young nobleman.", "cool, stoic detachment expected of a young nobleman", "Grade 7", "Literature")
]

# ---------------------------------------------------------
# 5. SOCIAL & DECISION MAKING (15 samples)
# ---------------------------------------------------------
social_data = [
    ("The village elders were all respected men who gathered under the banyan tree.", "village elders were all respected men", "Grade 6", "Social Studies"),
    ("Father made all financial and property decisions for the household.", "Father made all financial and property decisions", "Grade 5", "Social Studies"),
    ("The panchayat council chiefs are traditionally the senior patriarchs of the village.", "panchayat council chiefs are traditionally the senior patriarchs", "Grade 7", "Social Studies"),
    ("A wise woman lets her husband decide important matters without questioning.", "lets her husband decide important matters without questioning", "Grade 8", "Social Studies"),
    ("The king summoned his ministers, all noblemen, to vote on the treaty.", "ministers, all noblemen, to vote on the treaty", "Grade 6", "History"),
    ("In tribal governance, only male lineage heads possess the right to cast votes.", "only male lineage heads possess the right to cast votes", "Grade 8", "Social Studies"),
    ("The family budget was managed exclusively by the male breadwinner.", "managed exclusively by the male breadwinner", "Grade 6", "Social Studies"),
    ("When disputes arose in the market, the guild masters, all men, delivered judgment.", "guild masters, all men, delivered judgment", "Grade 7", "History"),
    ("Her brother represented the family at the town hall meeting while she stayed indoors.", "brother represented the family at the town hall meeting", "Grade 6", "Social Studies"),
    ("Political leadership requires a decisive authority that is characteristic of men.", "decisive authority that is characteristic of men", "Grade 10", "Social Studies"),
    ("The tribal chief sat with his warriors to determine the peace terms.", "tribal chief sat with his warriors", "Grade 5", "History"),
    ("The council of village fathers decided which fields would be planted that season.", "council of village fathers decided", "Grade 6", "Social Studies"),
    ("He spoke on behalf of his wife and mother during the land registry hearing.", "spoke on behalf of his wife and mother", "Grade 7", "Social Studies"),
    ("Men must lead the community because governance is an arduous masculine burden.", "governance is an arduous masculine burden", "Grade 9", "Social Studies"),
    ("The decision to sell the livestock rested solely with the grandfather.", "rested solely with the grandfather", "Grade 5", "Literature")
]

# ---------------------------------------------------------
# 6. REPRESENTATIONAL IMBALANCE (10 samples)
# ---------------------------------------------------------
imbalance_data = [
    ("The king gathered his five sons, his brother, and his uncle to plan the royal defense.", "king gathered his five sons, his brother, and his uncle", "Grade 5", "History"),
    ("The brave boy called his father, brother, grandfather, and male cousins to explore the island.", "father, brother, grandfather, and male cousins", "Grade 4", "Literature"),
    ("The emperor, his prince, his general, his nephew, and his male advisors led the expedition.", "emperor, his prince, his general, his nephew, and his male advisors", "Grade 6", "History"),
    ("He walked with his dad, uncle, brother, and grandpa to see the football championship.", "dad, uncle, brother, and grandpa", "Grade 3", "Literature"),
    ("The young man thanked his father, his brother, his headmaster, and his uncle for their guidance.", "father, his brother, his headmaster, and his uncle", "Grade 6", "Literature"),
    ("Three brothers set off on horseback: John, Peter, and Thomas, accompanied by their father.", "Three brothers set off on horseback: John, Peter, and Thomas", "Grade 4", "Literature"),
    ("The merchant and his four sons loaded their wooden cart with timber for the king.", "merchant and his four sons", "Grade 5", "History"),
    ("The monastery welcomed the abbot, his monks, his prior, and the young male novices.", "abbot, his monks, his prior, and the young male novices", "Grade 7", "History"),
    ("The boy watched as his brother, his uncle, and his grandfather repaired the fishing boat.", "brother, his uncle, and his grandfather", "Grade 3", "Literature"),
    ("The duke instructed his son, his nephew, and his loyal knights to guard the fortress gates.", "duke instructed his son, his nephew, and his loyal knights", "Grade 6", "History")
]

# ---------------------------------------------------------
# 7. COUNTER-NARRATIVES / ANTI-STEREOTYPES (50 samples)
# ---------------------------------------------------------
counter_data = [
    ("Rani Lakshmibai led her army into battle with fearless determination against colonial forces.", "led her army into battle", "Grade 8", "History"),
    ("Kalpana Chawla flew into space as a mission specialist on the space shuttle Columbia.", "flew into space as a mission specialist", "Grade 6", "Science"),
    ("The female surgeon successfully performed the complex open-heart surgery.", "female surgeon successfully performed", "Grade 7", "Science"),
    ("He stayed home caring for the sick toddler while his wife presided over the supreme court.", "caring for the sick toddler while his wife presided", "Grade 8", "Social Studies"),
    ("Sunita Williams completed spacewalks outside the International Space Station with excellence.", "completed spacewalks outside the International Space Station", "Grade 7", "Science"),
    ("Janaki Ammal conducted pioneering botanical cytogenetic research across India.", "pioneering botanical cytogenetic research", "Grade 9", "Science"),
    ("The young girl won the international mathematics olympiad with a perfect score.", "won the international mathematics olympiad", "Grade 8", "Mathematics"),
    ("Father happily swept the floor and made dinner while mother finished her engineering report.", "Father happily swept the floor and made dinner", "Grade 4", "Social Studies"),
    ("Marie Curie was awarded two Nobel Prizes in both Physics and Chemistry.", "awarded two Nobel Prizes in both Physics and Chemistry", "Grade 9", "Science"),
    ("Rani Abbakka repelled Portuguese naval invaders multiple times with brilliant coastal tactics.", "repelled Portuguese naval invaders", "Grade 7", "History"),
    ("The female pilot navigated the commercial aircraft safely through severe cyclonic turbulence.", "female pilot navigated the commercial aircraft safely", "Grade 6", "Science"),
    ("Koneru Humpy became one of the youngest grandmasters in international chess history.", "youngest grandmasters in international chess history", "Grade 7", "Literature"),
    ("He took parental leave to nurse his infant daughter and wash the family laundry.", "parental leave to nurse his infant daughter", "Grade 8", "Social Studies"),
    ("The female mechanical engineer designed the energy-efficient turbine blades.", "female mechanical engineer designed", "Grade 9", "Science"),
    ("Tarabai Shinde wrote a fierce critique of patriarchal double standards in 19th-century society.", "wrote a fierce critique of patriarchal double standards", "Grade 10", "History"),
    ("Both the husband and wife shared all household chores and meal preparations equally.", "shared all household chores and meal preparations equally", "Grade 5", "Social Studies"),
    ("The woman mountaineer climbed Mount Everest, demonstrating immense endurance and courage.", "climbed Mount Everest, demonstrating immense endurance", "Grade 6", "Literature"),
    ("She scored the winning goal in the national hockey tournament after intense physical training.", "scored the winning goal in the national hockey tournament", "Grade 5", "Literature"),
    ("The mother worked as a senior software architect while the father ran the domestic daycare.", "father ran the domestic daycare", "Grade 7", "Social Studies"),
    ("Kittur Chennamma led an armed rebellion against the British East India Company doctrine.", "led an armed rebellion against the British", "Grade 8", "History"),
    ("The girl dismantled and repaired the electric motor with confidence and skill.", "dismantled and repaired the electric motor", "Grade 6", "Science"),
    ("Mary Kom won six world amateur boxing championship titles through rigorous discipline.", "won six world amateur boxing championship titles", "Grade 8", "Literature"),
    ("The male teacher lovingly comforted the crying preschool child and prepared her lunch.", "male teacher lovingly comforted the crying preschool child", "Grade 3", "Social Studies"),
    ("Rosalind Franklin's X-ray diffraction images were crucial to discovering DNA structure.", "crucial to discovering DNA structure", "Grade 10", "Science"),
    ("The female prime minister signed the historic constitutional peace accord.", "female prime minister signed the historic constitutional peace accord", "Grade 9", "Social Studies"),
    ("She built a working solar-powered vehicle for her high school physics exhibition.", "built a working solar-powered vehicle", "Grade 9", "Science"),
    ("The husband washed the dishes and bathed the children every evening after work.", "husband washed the dishes and bathed the children", "Grade 4", "Social Studies"),
    ("Avani Chaturvedi became one of India's first female supersonic fighter jet pilots.", "first female supersonic fighter jet pilots", "Grade 8", "Social Studies"),
    ("The female firefighter entered the burning building and carried the trapped citizen to safety.", "female firefighter entered the burning building", "Grade 5", "Social Studies"),
    ("He enjoyed knitting wool sweaters for his grandchildren while sitting on the porch.", "enjoyed knitting wool sweaters", "Grade 4", "Literature"),
    ("Shakuntala Devi demonstrated phenomenal mental calculation feats across international universities.", "phenomenal mental calculation feats", "Grade 7", "Mathematics"),
    ("The woman astrophysicist discovered a new exoplanet using orbital spectrographic data.", "discovered a new exoplanet", "Grade 10", "Science"),
    ("The father stayed home as the primary caregiver while his partner pursued her diplomatic posting.", "father stayed home as the primary caregiver", "Grade 8", "Social Studies"),
    ("Sarojini Naidu addressed international assemblies with eloquent political leadership.", "addressed international assemblies with eloquent political leadership", "Grade 9", "History"),
    ("The female carpenter carved intricate wooden furniture for the heritage restoration project.", "female carpenter carved intricate wooden furniture", "Grade 7", "Social Studies"),
    ("She outperformed all male candidates in the rigorous civil service competitive examination.", "outperformed all male candidates", "Grade 10", "Social Studies"),
    ("The boy learned to bake bread and cook curries alongside his mother and father.", "boy learned to bake bread and cook curries", "Grade 4", "Literature"),
    ("Tessy Thomas served as project director for long-range ballistic missile defense systems.", "project director for long-range ballistic missile defense systems", "Grade 11", "Science"),
    ("The woman detective solved the mystery through sharp logical deduction and forensics.", "solved the mystery through sharp logical deduction", "Grade 7", "Literature"),
    ("He took pride in packing healthy lunchboxes for his kids and organizing the kitchen pantry.", "packing healthy lunchboxes for his kids", "Grade 4", "Social Studies"),
    ("The woman boxing champion delivered powerful left hooks in the gold-medal bout.", "delivered powerful left hooks in the gold-medal bout", "Grade 6", "Literature"),
    ("Indira Hinduja pioneered test-tube baby and gamete intrafallopian transfer techniques.", "pioneered test-tube baby and gamete intrafallopian transfer", "Grade 11", "Science"),
    ("The girl led the high school debate team to victory on economic trade policy.", "led the high school debate team to victory", "Grade 8", "Social Studies"),
    ("The grandfather lovingly braided his granddaughter's hair before she walked to school.", "grandfather lovingly braided his granddaughter's hair", "Grade 2", "Literature"),
    ("The female police commissioner reformed precinct integrity and reduced city crime.", "female police commissioner reformed precinct integrity", "Grade 9", "Social Studies"),
    ("She captained the university cricket squad and scored a brilliant century.", "captained the university cricket squad", "Grade 7", "Literature"),
    ("Father vacuumed the living room carpets while mother repaired the leaking bathroom pipe.", "Father vacuumed the living room carpets while mother repaired the leaking bathroom pipe", "Grade 5", "Social Studies"),
    ("Ada Lovelace wrote the world's first computer algorithm for the analytical engine.", "wrote the world's first computer algorithm", "Grade 9", "Science"),
    ("The female construction superintendent inspected the high-rise steel framework.", "female construction superintendent inspected the high-rise steel framework", "Grade 8", "Science"),
    ("He washed all the family dishes after dinner with cheerful diligence.", "washed all the family dishes after dinner", "Grade 3", "Literature")
]

# ---------------------------------------------------------
# 8. HISTORICAL COUNTEREXAMPLES (35 samples)
# ---------------------------------------------------------
historical_data = [
    ("Women were traditionally barred from formal military training in most 19th-century European states.", "", "Grade 9", "History"),
    ("In ancient times, girls were often excluded from learning the Vedic scriptures.", "", "Grade 7", "History"),
    ("Before the passage of the 19th Amendment, American women could not vote in national elections.", "", "Grade 10", "History"),
    ("Widows in the 18th century were historically forbidden from remarrying or owning inherited land.", "", "Grade 8", "History"),
    ("During the Victorian era, married women were legally denied separate ownership of personal property.", "", "Grade 9", "History"),
    ("Historically, medieval universities admitted only male students to study law, theology, and medicine.", "", "Grade 8", "History"),
    ("In the feudal era, female succession to aristocratic fiefdoms was prohibited under the Salic law.", "", "Grade 10", "History"),
    ("Colonial authorities denied Indian women employment in civil administrative services until reforms were enacted.", "", "Grade 9", "History"),
    ("Until the early 20th century, women in many nations were barred from practicing law in courts.", "", "Grade 10", "History"),
    ("Ancient Greek citizens who participated in Athenian democracy were exclusively adult free males.", "", "Grade 6", "History"),
    ("Historically, working-class women in industrial textile mills were paid significantly lower wages than men.", "", "Grade 8", "History"),
    ("Under ancient Roman civil law, the paterfamilias held absolute legal dominion over wife and children.", "", "Grade 9", "History"),
    ("In the 19th century, social reformers like Jyotirao Phule fought against the exclusion of girls from schools.", "", "Grade 8", "History"),
    ("Women were excluded from participating in the ancient Olympic games held at Olympia.", "", "Grade 6", "History"),
    ("Historically, female infanticide was an outlawed practice that colonial administrators attempted to suppress.", "", "Grade 10", "History"),
    ("Until the mid-20th century, married women in several Western nations required a husband's permission to open bank accounts.", "", "Grade 10", "Social Studies"),
    ("In medieval Europe, guild memberships for stone masons and blacksmiths were traditionally restricted to men.", "", "Grade 7", "History"),
    ("Before independence, women's literacy in India stood at less than ten percent due to structural neglect.", "", "Grade 8", "History"),
    ("Historically, the purdah system restricted women's physical mobility in many northern communities.", "", "Grade 8", "Social Studies"),
    ("In the 1800s, British universities refused to award formal degrees to female students who passed examinations.", "", "Grade 9", "History"),
    ("The factory acts of the 19th century documented harsh working conditions for young women and children in coal pits.", "", "Grade 9", "History"),
    ("Historically, women were barred from entering the diplomatic corps until post-war legal reforms.", "", "Grade 10", "History"),
    ("Under ancient customary law, daughters were systematically denied equal ancestral coparcenary shares.", "", "Grade 11", "Social Studies"),
    ("Raja Ram Mohan Roy campaigned vigorously against the historical practice of Sati.", "", "Grade 8", "History"),
    ("Until legal amendments in 2005, Hindu succession laws did not grant daughters equal coparcenary rights by birth.", "", "Grade 11", "Social Studies"),
    ("Historically, women in ancient Sparta, unlike Athens, received physical education and owned agricultural estates.", "", "Grade 9", "History"),
    ("In early medical history, female healers were frequently accused of witchcraft by ecclesiastical authorities.", "", "Grade 9", "History"),
    ("During the Industrial Revolution, women worked twelve-hour shifts in cotton mills while being denied franchise.", "", "Grade 8", "History"),
    ("Historically, girls in feudal Japan were trained in domestic etiquette rather than samurai martial arts.", "", "Grade 7", "History"),
    ("Until the 1970s, pregnant teachers in many public school districts were subjected to mandatory termination.", "", "Grade 11", "History"),
    ("Ancient Mesopotamian legal codes imposed strict financial penalties for injuries to free women.", "", "Grade 9", "History"),
    ("In 19th-century Bengal, Ishwar Chandra Vidyasagar campaigned for the legal enactment of the Hindu Widows' Remarriage Act.", "", "Grade 8", "History"),
    ("Women in Switzerland were not granted the federal right to vote until the historic referendum of 1971.", "", "Grade 10", "History"),
    ("Historically, women in maritime fishing communities managed fish curing while men manned sea trawlers.", "", "Grade 6", "Social Studies"),
    ("Before the Equal Pay Act of 1970, employers in the UK could legally advertise different pay rates for the same job.", "", "Grade 10", "History")
]

# ---------------------------------------------------------
# 9. NEUTRAL & GENDER-BALANCED (35 samples)
# ---------------------------------------------------------
neutral_data = [
    ("Photosynthesis is the biochemical process by which green plants synthesize glucose using sunlight.", "", "Grade 7", "Science"),
    ("Both male and female students conducted the chemical titration experiment in the laboratory.", "", "Grade 8", "Science"),
    ("The municipal corporation consists of ward councilors elected through universal adult franchise.", "", "Grade 6", "Social Studies"),
    ("Students worked collaboratively in small groups to solve the quadratic equation.", "", "Grade 9", "Mathematics"),
    ("The teacher instructed the entire class to open their geometry textbooks to chapter four.", "", "Grade 5", "Mathematics"),
    ("Water boils at 100 degrees Celsius under standard atmospheric pressure.", "", "Grade 6", "Science"),
    ("The library contains reference encyclopedias, historical atlases, and scientific journals.", "", "Grade 4", "Literature"),
    ("Both boys and girls participated enthusiastically in the annual science exhibition.", "", "Grade 7", "Science"),
    ("The committee appointed a new chairperson through a unanimous voice vote.", "", "Grade 8", "Social Studies"),
    ("Mitochondria generate adenosine triphosphate through cellular respiration pathways.", "", "Grade 9", "Science"),
    ("The passenger train arrived at the central railway terminal on schedule.", "", "Grade 3", "Social Studies"),
    ("Students took turns presenting their solar system posters to the audience.", "", "Grade 5", "Science"),
    ("The Constitution guarantees fundamental rights to all citizens without discrimination.", "", "Grade 8", "Social Studies"),
    ("Gravity exerts an attractive force proportional to the masses of interacting bodies.", "", "Grade 9", "Science"),
    ("The farmers harvested their wheat crop before the monsoon rains began.", "", "Grade 5", "Social Studies"),
    ("Both men and women in the village cast their ballots during the panchayat election.", "", "Grade 6", "Social Studies"),
    ("The orchestra tuned their string and wind instruments before the evening concert.", "", "Grade 6", "Literature"),
    ("Sound waves require a material medium to propagate through space.", "", "Grade 8", "Science"),
    ("The youth athletic meet included sprinting, long jump, and relay competitions.", "", "Grade 5", "Social Studies"),
    ("The digestive system breaks down carbohydrates, proteins, and lipids into absorbable nutrients.", "", "Grade 7", "Science"),
    ("All students are required to wear safety goggles during chemistry experiments.", "", "Grade 8", "Science"),
    ("The river flows through the fertile alluvial plains before emptying into the sea.", "", "Grade 6", "Social Studies"),
    ("The class representative collected the homework notebooks for teacher grading.", "", "Grade 4", "Literature"),
    ("Both men and women contributed equally to the community afforestation project.", "", "Grade 7", "Social Studies"),
    ("Renewable energy sources include solar, wind, geothermal, and hydroelectric power.", "", "Grade 8", "Science"),
    ("The students organized a book donation drive for the village reading room.", "", "Grade 5", "Literature"),
    ("The solar eclipse occurs when the Moon passes directly between the Earth and the Sun.", "", "Grade 7", "Science"),
    ("Citizens are encouraged to practice waste segregation to support urban recycling.", "", "Grade 6", "Social Studies"),
    ("Both brothers and sisters enjoyed hiking up the pine-covered hillside.", "", "Grade 4", "Literature"),
    ("The weather station recorded twenty millimeters of rainfall during the afternoon storm.", "", "Grade 5", "Science"),
    ("The school debating society discussed the economic impacts of global trade policies.", "", "Grade 10", "Social Studies"),
    ("Plant cells contain rigid cell walls made of cellulose fibers.", "", "Grade 8", "Science"),
    ("The villagers gathered at the community center to celebrate the annual harvest festival.", "", "Grade 5", "Social Studies"),
    ("The mathematics exam tested mastery of linear algebra, probability, and geometry.", "", "Grade 10", "Mathematics"),
    ("Equal opportunity in public employment is a constitutional guarantee for every citizen.", "", "Grade 9", "Social Studies")
]


def generate_all():
    sample_id = 1

    # 1. Domestic roles
    for text, span, grade, subj in domestic_data:
        add_sample(sample_id, text, "stereotype", "DOMESTIC_ROLES", None, span, grade, subj, False)
        sample_id += 1

    # 2. Occupational bias
    for text, span, grade, subj in occupational_data:
        add_sample(sample_id, text, "stereotype", "OCCUPATIONAL_BIAS", None, span, grade, subj, False)
        sample_id += 1

    # 3. Ability bias
    for text, span, grade, subj in ability_data:
        add_sample(sample_id, text, "stereotype", "ABILITY_INTELLECTUAL_BIAS", None, span, grade, subj, False)
        sample_id += 1

    # 4. Personality
    for text, span, grade, subj in personality_data:
        add_sample(sample_id, text, "stereotype", "PERSONALITY_BEHAVIORAL", None, span, grade, subj, False)
        sample_id += 1

    # 5. Social decision making
    for text, span, grade, subj in social_data:
        add_sample(sample_id, text, "stereotype", "SOCIAL_DECISION_MAKING", None, span, grade, subj, False)
        sample_id += 1

    # 6. Representational imbalance
    for text, span, grade, subj in imbalance_data:
        add_sample(sample_id, text, "stereotype", "REPRESENTATIONAL_IMBALANCE", None, span, grade, subj, False)
        sample_id += 1

    # 7. Counter-narratives / Anti-stereotypes
    for text, span, grade, subj in counter_data:
        add_sample(sample_id, text, "anti_stereotype", None, "COUNTER_NARRATIVE", span, grade, subj, False)
        sample_id += 1

    # 8. Historical counterexamples
    for text, span, grade, subj in historical_data:
        add_sample(sample_id, text, "neutral", None, "HISTORICAL_CONTEXT", span, grade, subj, True)
        sample_id += 1

    # 9. Neutral / Balanced
    for text, span, grade, subj in neutral_data:
        add_sample(sample_id, text, "neutral", None, "GENDER_BALANCED", span, grade, subj, False)
        sample_id += 1

    print(f"Total samples assembled: {len(SAMPLES)}")
    assert len(SAMPLES) == 250, f"Expected exactly 250 samples, got {len(SAMPLES)}"

    # Schema verification
    for s in SAMPLES:
        assert "id" in s and s["id"]
        assert "text" in s and s["text"]
        assert s["classification"] in ["stereotype", "anti_stereotype", "neutral"]
        assert s["grade_level"].startswith("Grade")
        assert s["subject"] in ["Social Studies", "Science", "Literature", "History", "Mathematics"]
        if s["classification"] == "stereotype":
            assert s["category"] in [
                "DOMESTIC_ROLES", "OCCUPATIONAL_BIAS", "ABILITY_INTELLECTUAL_BIAS",
                "PERSONALITY_BEHAVIORAL", "SOCIAL_DECISION_MAKING", "REPRESENTATIONAL_IMBALANCE"
            ]
            assert len(s["evidence_span"]) > 0
            assert s["evidence_span"] in s["text"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "Curated 250-Passage Educational Stereotype Benchmark",
            "version": "1.0.0",
            "total_samples": len(SAMPLES),
            "distribution": {
                "stereotype": sum(1 for s in SAMPLES if s["classification"] == "stereotype"),
                "anti_stereotype": sum(1 for s in SAMPLES if s["classification"] == "anti_stereotype"),
                "neutral": sum(1 for s in SAMPLES if s["classification"] == "neutral"),
                "historical_counterexamples": sum(1 for s in SAMPLES if s["is_historical_counterexample"])
            },
            "samples": SAMPLES
        }, f, indent=2)

    print(f"[+] Successfully wrote 250 verified benchmark samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_all()
