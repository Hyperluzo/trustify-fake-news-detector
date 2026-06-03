"""
Fact Checker v3.0 — Comprehensive World Knowledge Verification
Catches: short names, nicknames, CEO facts, science basics,
         question format ("Is X PM of Y?"), any word order
"""
import re
from typing import Dict, List, Tuple

# ── WORLD LEADERS: short name + full name + nicknames ─────────────────────
# Format: (list_of_name_variants, correct_country, correct_role)
LEADERS = [
    # India
    (['narendra modi','modi'],                    'India',         'Prime Minister'),
    (['droupadi murmu','murmu'],                  'India',         'President'),
    (['amit shah'],                               'India',         'Home Minister'),
    (['rahul gandhi'],                            'India',         'Congress Leader'),
    (['manmohan singh'],                          'India',         'former Prime Minister'),
    # USA
    (['donald trump','trump'],                    'United States', 'former President'),
    (['joe biden','biden'],                       'United States', 'President'),
    (['kamala harris','harris'],                  'United States', 'Vice President'),
    (['barack obama','obama'],                    'United States', 'former President'),
    (['george bush','bush'],                      'United States', 'former President'),
    (['bill clinton','clinton'],                  'United States', 'former President'),
    # UK
    (['rishi sunak','sunak'],                    'United Kingdom','Prime Minister'),
    (['keir starmer','starmer'],                 'United Kingdom','Prime Minister'),
    (['boris johnson','boris'],                  'United Kingdom','former Prime Minister'),
    (['king charles','charles iii'],             'United Kingdom','King'),
    (['queen elizabeth','queen elizabeth ii'],   'United Kingdom','former Queen'),
    # Russia
    (['vladimir putin','putin'],                 'Russia',        'President'),
    # China
    (['xi jinping','xi'],                        'China',         'President'),
    # Pakistan
    (['shehbaz sharif','shehbaz'],              'Pakistan',      'Prime Minister'),
    (['imran khan','imran'],                     'Pakistan',      'former Prime Minister'),
    # France
    (['emmanuel macron','macron'],              'France',        'President'),
    # Germany
    (['olaf scholz','scholz'],                  'Germany',       'Chancellor'),
    (['angela merkel','merkel'],                'Germany',       'former Chancellor'),
    # Canada
    (['justin trudeau','trudeau'],              'Canada',        'Prime Minister'),
    # Australia
    (['anthony albanese','albanese'],           'Australia',     'Prime Minister'),
    # Japan
    (['fumio kishida','kishida'],               'Japan',         'former Prime Minister'),
    # Italy
    (['giorgia meloni','meloni'],               'Italy',         'Prime Minister'),
    # Brazil
    (['luiz lula','lula'],                      'Brazil',        'President'),
    # Israel
    (['benjamin netanyahu','netanyahu','bibi'], 'Israel',        'Prime Minister'),
    # Turkey
    (['recep erdogan','erdogan'],              'Turkey',         'President'),
    # South Africa
    (['cyril ramaphosa','ramaphosa'],          'South Africa',   'President'),
    # Ukraine
    (['volodymyr zelensky','zelensky'],        'Ukraine',        'President'),
    # North Korea
    (['kim jong un','kim jong-un'],            'North Korea',    'Supreme Leader'),
]

# ── COUNTRIES to check against ─────────────────────────────────────────────
ALL_COUNTRIES = [
    'india','indian','usa','america','american','united states','us',
    'uk','britain','british','england','united kingdom',
    'china','chinese','russia','russian','france','french',
    'germany','german','pakistan','pakistani','australia','australian',
    'canada','canadian','japan','japanese','italy','italian',
    'brazil','brazilian','saudi arabia','uae','iran','iraqi','iraq',
    'turkey','turkish','israel','israeli','ukraine','ukrainian',
    'south africa','north korea','korean','nepal','bangladesh',
    'mexico','spanish','spain','argentina','egypt','nigeria','kenya',
]

# ── COMPANY / TECH FACTS ─────────────────────────────────────────────────
# (wrong_claim_variants, truth)
COMPANY_FACTS = [
    # ChatGPT / OpenAI
    (['chatgpt made by google','chatgpt is google','chatgpt by google',
      'chatgpt created by google','chatgpt built by google','google made chatgpt',
      'google created chatgpt','chatgpt is from google'],
     'ChatGPT was created by OpenAI, not Google. Google makes Gemini/Bard.'),
    (['openai owned by google','openai is google','google owns openai'],
     'OpenAI is an independent company, not owned or made by Google.'),
    (['gemini made by openai','bard made by openai','google ai made by openai'],
     'Gemini (formerly Bard) is Google\'s AI, made by Google — not OpenAI.'),
    (['chatgpt made by microsoft','chatgpt is microsoft','microsoft created chatgpt'],
     'ChatGPT was created by OpenAI. Microsoft is an investor in OpenAI but did not create it.'),
    (['elon musk owns openai','musk owns openai','elon musk created chatgpt'],
     'Elon Musk co-founded OpenAI in 2015 but left in 2018. He does not own or run OpenAI.'),
    # Elon Musk
    (['elon musk ceo of apple','musk is ceo of apple','musk runs apple'],
     'Elon Musk is not CEO of Apple. Tim Cook is CEO of Apple. Musk runs Tesla, SpaceX, and X.'),
    (['elon musk ceo of google','musk ceo google'],
     'Elon Musk is not CEO of Google. Sundar Pichai is CEO of Google/Alphabet.'),
    (['elon musk ceo of amazon','musk ceo amazon'],
     'Elon Musk is not CEO of Amazon. Andy Jassy is CEO of Amazon.'),
    (['elon musk ceo of meta','musk ceo facebook'],
     'Elon Musk is not CEO of Meta/Facebook. Mark Zuckerberg is CEO of Meta.'),
    # Apple
    (['steve jobs ceo of apple','steve jobs runs apple'],
     'Steve Jobs passed away in 2011. Tim Cook has been CEO of Apple since then.'),
    (['tim cook ceo of google','tim cook ceo of microsoft'],
     'Tim Cook is CEO of Apple, not Google or Microsoft.'),
    # Amazon
    (['jeff bezos ceo of amazon','bezos runs amazon','bezos is ceo'],
     'Jeff Bezos stepped down as Amazon CEO in 2021. Andy Jassy is now CEO of Amazon.'),
    # Microsoft
    (['bill gates ceo of microsoft','bill gates runs microsoft'],
     'Bill Gates stepped down as Microsoft CEO in 2000. Satya Nadella has been CEO since 2014.'),
    (['elon musk ceo of microsoft'],
     'Elon Musk is not CEO of Microsoft. Satya Nadella is CEO of Microsoft.'),
    # Facebook/Meta
    (['elon musk ceo of facebook','musk owns facebook'],
     'Elon Musk does not own Facebook/Meta. Mark Zuckerberg is CEO and founder of Meta.'),
    (['mark zuckerberg owns twitter','zuckerberg runs twitter'],
     'Mark Zuckerberg does not own Twitter/X. Elon Musk acquired Twitter in 2022, now called X.'),
]

# ── CAPITAL CITIES ────────────────────────────────────────────────────────
WRONG_CAPITALS = [
    (['capital of india is mumbai','capital of india is delhi','capital india mumbai',
      'india capital mumbai','capital of india mumbai'],
     'The capital of India is New Delhi, not Mumbai.'),
    (['capital of australia is sydney','capital australia sydney','sydney is capital of australia'],
     'The capital of Australia is Canberra, not Sydney.'),
    (['capital of usa is new york','capital of america is new york','new york is capital of usa',
      'new york capital america','us capital new york'],
     'The capital of the USA is Washington D.C., not New York.'),
    (['capital of canada is toronto','toronto is capital of canada'],
     'The capital of Canada is Ottawa, not Toronto.'),
    (['capital of brazil is sao paulo','sao paulo capital brazil'],
     'The capital of Brazil is Brasília, not São Paulo.'),
    (['capital of pakistan is lahore','lahore capital pakistan'],
     'The capital of Pakistan is Islamabad, not Lahore.'),
    (['capital of china is shanghai','shanghai capital china'],
     'The capital of China is Beijing, not Shanghai.'),
    (['capital of new zealand is auckland','auckland capital new zealand'],
     'The capital of New Zealand is Wellington, not Auckland.'),
    (['capital of uk is manchester','capital of england is manchester',
      'capital of britain is birmingham'],
     'The capital of the UK is London.'),
    (['capital of japan is osaka','osaka capital japan'],
     'The capital of Japan is Tokyo, not Osaka.'),
    (['capital of france is nice','capital of france is lyon'],
     'The capital of France is Paris.'),
    (['capital of germany is munich','munich capital germany'],
     'The capital of Germany is Berlin, not Munich.'),
    (['capital of russia is st petersburg'],
     'The capital of Russia is Moscow, not St. Petersburg.'),
    (['capital of spain is barcelona','barcelona capital spain'],
     'The capital of Spain is Madrid, not Barcelona.'),
    (['capital of italy is milan','milan capital italy'],
     'The capital of Italy is Rome, not Milan.'),
]

# ── SCIENCE FACTS ─────────────────────────────────────────────────────────
SCIENCE_FACTS = [
    # Sun/Earth
    (['sun revolves around earth','earth is center of solar system',
      'sun goes around earth','earth does not revolve around sun',
      'sun orbits earth','earth is center of universe'],
     'The Earth revolves around the Sun, not the other way around. This has been proven for 500 years.'),
    (['earth is flat','flat earth','earth is not round','earth is flat confirmed'],
     'Earth is a sphere (oblate spheroid). This is proven by space imagery, GPS, and physics.'),
    # Gravity
    (['einstein discovered gravity','einstein invented gravity',
      'einstein found gravity','gravity discovered by einstein'],
     'Gravity was described by Isaac Newton in 1687. Einstein developed the theory of relativity.'),
    (['newton discovered relativity','newton theory of relativity'],
     'The theory of relativity was developed by Albert Einstein, not Newton.'),
    # Moon
    (['moon is made of cheese','moon made of cheese'],
     'The moon is made of rock and regolith, not cheese.'),
    (['moon landing faked','moon landing was fake','nasa faked moon landing'],
     'The Apollo moon landings were real. They are among the most documented events in history.'),
    # Water
    (['water boils at 50','water boils at 90 degree','water boils at 200'],
     'Water boils at 100°C (212°F) at standard atmospheric pressure at sea level.'),
    # Brain
    (['humans use 10 percent of brain','we use only 10 percent',
      'humans only use 10% of their brain'],
     'Humans use virtually all of their brain. The 10% myth has been thoroughly debunked by neuroscience.'),
    # Great Wall
    (['great wall of china visible from space','great wall visible from moon'],
     'The Great Wall of China is NOT visible from space with the naked eye. NASA astronauts confirmed this.'),
    # Lightning
    (['lightning never strikes twice'],
     'Lightning can and does strike the same place multiple times.'),
    # Vaccines
    (['vaccines cause autism','vaccine causes autism'],
     'Extensive research across millions of children has found NO link between vaccines and autism.'),
    # COVID
    (['covid vaccine has microchip','vaccine microchip','bill gates microchip vaccine'],
     'COVID vaccines do not contain microchips. This has been thoroughly fact-checked and debunked.'),
    (['drinking bleach cures covid','bleach cures coronavirus'],
     'Drinking bleach is extremely dangerous and potentially fatal. It does not cure any disease.'),
    (['5g causes covid','5g towers spread covid','5g and coronavirus'],
     '5G technology does not cause or spread COVID-19. They are completely unrelated.'),
    # Oxygen
    (['oxygen is blue','oxygen is red gas'],
     'Oxygen is a colorless, odorless gas at room temperature.'),
    # Sun
    (['sun rises in west','sun rises from west'],
     'The sun rises in the East and sets in the West. This is basic astronomy.'),
    (['sun sets in east'],
     'The sun sets in the West, not the East.'),
    # Blood
    (['blood is blue inside body','blood blue veins'],
     'Human blood is always red. Veins look blue through skin due to light absorption, not blood color.'),
]

# ── HISTORY FACTS ─────────────────────────────────────────────────────────
HISTORY_FACTS = [
    (['india independence 1948','india independence 1949','india independence 1946',
      'india got independence 1950','india independence in 1950'],
     'India gained independence on August 15, 1947, not any other year.'),
    (['world war 2 ended 1943','world war 2 ended 1944','ww2 ended 1946','ww2 ended 1950'],
     'World War 2 ended in 1945 with Allied victory.'),
    (['world war 1 ended 1920','ww1 ended 1919','ww1 ended 1920'],
     'World War 1 ended on November 11, 1918.'),
    (['mahatma gandhi first prime minister','gandhi was prime minister india'],
     'India\'s first Prime Minister was Jawaharlal Nehru, not Mahatma Gandhi.'),
    (['nehru first president india','nehru was president'],
     'Nehru was India\'s first Prime Minister. Dr. Rajendra Prasad was the first President.'),
    (['us independence 1775','america independence 1777','us independence 1780'],
     'The USA declared independence on July 4, 1776.'),
    (['columbus discovered india','columbus reached india'],
     'Columbus reached the Americas in 1492, not India.'),
    (['titanic sank 1910','titanic sank 1913','titanic sank 1915'],
     'The Titanic sank on April 15, 1912.'),
    (['chatgpt was made by google'],  # duplicate safety
     'ChatGPT was created by OpenAI, not Google.'),
    (['virat kohli 100 test centuries','kohli 100 test tons','kohli scored 100 centuries in test'],
     'Virat Kohli has NOT scored 100 test centuries. Sachin Tendulkar holds the record of 100 international centuries.'),
    (['sachin tendulkar 200 centuries','sachin 200 international centuries'],
     'Sachin Tendulkar scored 100 international centuries, not 200.'),
]

# ── SPORTS FACTS ──────────────────────────────────────────────────────────
SPORTS_FACTS = [
    (['virat kohli 100 test centuries','kohli 100 centuries in test cricket'],
     'Virat Kohli has NOT scored 100 test centuries. Sachin Tendulkar holds the record of 100 international centuries.'),
    (['messi won world cup 2018','messi 2018 world cup','argentina won 2018 world cup'],
     'Argentina/Messi did NOT win the 2018 World Cup. France won in 2018. Argentina won in 2022.'),
    (['brazil won 2022 world cup'],
     'Brazil did NOT win the 2022 World Cup. Argentina won the 2022 FIFA World Cup.'),
    (['india won cricket world cup 2023','india 2023 world cup winner'],
     'India did NOT win the 2023 ODI World Cup. Australia won the 2023 ODI World Cup.'),
    (['india won 2024 t20 world cup'],
     'India WON the 2024 T20 World Cup — this is TRUE, not fake.'),  # this one is REAL
]

# ── DIRECT FAKE PHRASES (exact match or near-match) ───────────────────────
DIRECT_FAKE = [
    # Trump as PM
    ('trump is pm of india',            'Donald Trump is former President of USA, not PM of India.'),
    ('trump pm india',                  'Donald Trump is former President of USA, not PM of India.'),
    ('trump prime minister india',      'Donald Trump is former President of USA, not PM of India.'),
    ('trump is prime minister of india','Donald Trump is former President of USA, not PM of India.'),
    ('is trump pm of india',            'Donald Trump is former President of USA, not PM of India.'),
    ('is trump prime minister of india','Donald Trump is former President of USA, not PM of India.'),
    ('trump became pm of india',        'Donald Trump is former President of USA, not PM of India.'),
    # Obama wrong roles
    ('obama pm of uk',                  'Barack Obama is former President of USA, not PM of UK.'),
    ('obama prime minister uk',         'Barack Obama is former President of USA, not PM of UK.'),
    ('obama prime minister of uk',      'Barack Obama is former President of USA, not PM of UK.'),
    ('is obama prime minister',         'Barack Obama is former President of USA, not a Prime Minister of any country.'),
    # Musk CEO Apple
    ('elon musk ceo apple',             'Tim Cook is CEO of Apple, not Elon Musk.'),
    ('is elon musk ceo of apple',       'Tim Cook is CEO of Apple, not Elon Musk.'),
    ('musk is ceo of apple',            'Tim Cook is CEO of Apple, not Elon Musk.'),
    # Sun/Earth
    ('does sun revolve around earth',   'No — Earth revolves around the Sun, not the other way.'),
    ('sun revolves around earth',       'Earth revolves around the Sun. This has been proven for 500+ years.'),
    ('is earth center of universe',     'Earth is not the center of the universe. There is no single center.'),
    # ChatGPT
    ('chatgpt google',                  'ChatGPT was made by OpenAI, not Google.'),
    ('chatgpt is made by google',       'ChatGPT was created by OpenAI, not Google.'),
    ('chatgpt made by google',          'ChatGPT was created by OpenAI, not Google.'),
    ('chatgpt from google',             'ChatGPT is from OpenAI, not Google.'),
    ('chatgpt created google',          'ChatGPT was created by OpenAI, not Google.'),
    ('chatgpt is google',               'ChatGPT is made by OpenAI, not Google.'),
    ('chatgpt belongs to google',       'ChatGPT belongs to OpenAI, not Google.'),
    ('is chatgpt made by google',       'No — ChatGPT was created by OpenAI. Google makes Gemini.'),
    ('india got independence in 1950',  'India gained independence on August 15, 1947, not 1950.'),
    ('india independence 1950',         'India gained independence on August 15, 1947, not 1950.'),
    ('india independence was in 1950',  'India gained independence on August 15, 1947, not 1950.'),
    ('independence of india 1950',      'India gained independence on August 15, 1947.'),
    ('chatgpt is google product',       'ChatGPT is an OpenAI product, not Google.'),
    # Basic wrong facts
    ('australia capital sydney',        'Capital of Australia is Canberra, not Sydney.'),
    ('sydney is capital',               'Sydney is NOT the capital of Australia. Canberra is.'),
    ('sydney is the capital of australia','The capital of Australia is Canberra, not Sydney.'),
    ('sydney capital of australia',     'The capital of Australia is Canberra, not Sydney.'),
    ('einstein discovered gravity',     'Gravity was described by Isaac Newton, not Einstein.'),
    ('sun rises in west',               'Sun rises in the East, not the West.'),
    ('earth is flat',                   'Earth is a sphere, proven by science and space exploration.'),
    ('aliens landed in delhi',          'No verified evidence of alien landings exists anywhere on Earth.'),
    ('aliens landed in',                'No verified evidence of alien landings exists anywhere on Earth.'),
    ('alien landed in',                 'No verified evidence of alien landings exists anywhere on Earth.'),
    ('bill gates microchip',            'COVID vaccines do not contain Bill Gates microchips — thoroughly debunked.'),
    ('microchip in covid vaccine',      'COVID vaccines do not contain microchips — this is debunked misinformation.'),
    ('microchip in vaccine',            'Vaccines do not contain microchips — this has been thoroughly debunked.'),
    ('5g towers controlling',             '5G technology cannot control human minds — this is misinformation.'),
    ('5g towers are controlling',          '5G technology cannot control human minds — this is debunked misinformation.'),
    ('5g is controlling',                  '5G technology does not control human minds — this is a conspiracy theory.'),
    ('5g controlling minds',               '5G technology cannot control human minds.'),
    ('5g mind control',                    '5G technology does not enable mind control — this is misinformation.'),
    ('5g towers mind control',             '5G technology does not enable mind control.'),
    ('5g controlling human',            '5G technology cannot control human minds — this is misinformation.'),
    ('government hiding aliens',        'Government cover-up of alien contact is a well-known conspiracy theory with no evidence.'),
    ('illuminati controls',             'The Illuminati controlling governments is a conspiracy theory with no factual basis.'),
    ('new world order',                 'New World Order is a conspiracy theory, not an established fact.'),
    ('chemtrail',                       'Chemtrails are a conspiracy theory — aircraft contrails are condensed water vapor.'),
    ('flat earth proven',               'Earth is a sphere — flat earth claims are debunked by science and space imagery.'),
]


def normalize(text: str) -> str:
    """Normalize text for matching: lowercase, collapse spaces, remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_leader_wrong_country(text_lower: str) -> List[str]:
    """Check if any world leader is mentioned with a wrong country."""
    contradictions = []
    for name_variants, correct_country, role in LEADERS:
        found_name = None
        for variant in name_variants:
            if variant in text_lower:
                found_name = variant
                break
        if not found_name:
            continue

        # If the CORRECT country is mentioned — this is TRUE, skip entirely
        correct_lower = correct_country.lower()
        correct_words = correct_lower.replace('united ', '').split()
        if any(cw in text_lower for cw in correct_words if len(cw) > 3):
            continue

        # If "former" + correct role mentioned — this is a TRUE historical fact, skip
        role_lower = role.lower().replace('former ', '')
        if 'former' in text_lower and role_lower in text_lower:
            continue

        # If "prime minister of uk" or "pm of uk" etc. AND name is Rishi/Sunak — TRUE, skip
        # General: if ANY correct country synonym appears near the name, skip
        country_synonyms = {
            'united states': ['usa', 'us', 'america', 'american', 'united states'],
            'united kingdom': ['uk', 'britain', 'british', 'england', 'united kingdom'],
            'india': ['india', 'indian'],
            'russia': ['russia', 'russian'],
            'china': ['china', 'chinese'],
            'france': ['france', 'french'],
            'germany': ['germany', 'german'],
            'pakistan': ['pakistan', 'pakistani'],
            'australia': ['australia', 'australian'],
            'canada': ['canada', 'canadian'],
        }
        synonyms = country_synonyms.get(correct_lower, [correct_lower])
        if any(syn in text_lower for syn in synonyms):
            continue

        name_pos = text_lower.find(found_name)
        # Check if any WRONG country appears near the name (within 120 chars)
        for country in ALL_COUNTRIES:
            if country in text_lower:
                country_pos = text_lower.find(country)
                if abs(name_pos - country_pos) < 120:
                    if country not in correct_lower and correct_lower not in country:
                        msg = f"{found_name.title()} is {role} of {correct_country}, not of {country.title()}"
                        if msg not in contradictions:
                            contradictions.append(msg)
    return contradictions


def check_company_facts(text_lower: str) -> List[str]:
    """Check for wrong company/tech attributions."""
    contradictions = []
    for patterns, truth in COMPANY_FACTS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in contradictions:
                    contradictions.append(truth)
                break
    return contradictions


def check_capitals(text_lower: str) -> List[str]:
    contradictions = []
    for patterns, truth in WRONG_CAPITALS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in contradictions:
                    contradictions.append(truth)
                break
    return contradictions


def check_science(text_lower: str) -> List[str]:
    import re as _re
    contradictions = []
    for patterns, truth in SCIENCE_FACTS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in contradictions:
                    contradictions.append(truth)
                break

    # Regex-based checks for flexible phrasing
    REGEX_SCIENCE = [
        (r'sun\s+revolves?\s+around\s+(the\s+)?earth',
         'Earth revolves around the Sun, not the other way. Proven for 500+ years.'),
        (r'earth\s+is\s+(the\s+)?center\s+of\s+(the\s+)?(universe|solar system)',
         'Earth is not the center of the universe or solar system.'),
        (r'(flat\s+earth|earth\s+is\s+flat)',
         'Earth is a sphere — proven by science and space exploration.'),
        (r'sun\s+(rises?|sets?)\s+(in|from)\s+the\s+west',
         'The sun rises in the East and sets in the West.'),
        (r'(einstein|albert einstein)\s+(discovered|invented|found)\s+gravity',
         'Gravity was described by Isaac Newton (1687). Einstein developed relativity.'),
        (r'lightning\s+never\s+strikes?\s+twice',
         'Lightning can and does strike the same place multiple times.'),
        (r'(vaccines?\s+cause[sd]?\s+autism|autism\s+(caused|from)\s+vaccine)',
         'No scientific link between vaccines and autism — debunked by large-scale studies.'),
        (r'(5g|five\s*g)\s+(cause[sd]?|spread[s]?|gives?)\s+covid',
         '5G technology does not cause or spread COVID-19.'),
        (r'(5g|five\s*g).{0,30}(mind\s+control|control.{0,10}mind)',
         '5G technology cannot control human minds — this is misinformation.'),
        (r'bill\s+gates.{0,30}microchip.{0,30}vaccine',
         'COVID vaccines do not contain Bill Gates microchips — thoroughly debunked.'),
        (r'microchip.{0,30}(covid\s+)?vaccine',
         'Vaccines do not contain microchips — this is debunked misinformation.'),
        (r'aliens?.{0,20}(landed|arrived|visited|met).{0,30}(delhi|india|mumbai|government)',
         'No verified evidence of alien landings exists on Earth.'),
        (r'(drink|drinking|consume).{0,20}bleach.{0,30}(cure|treat|fix)',
         'Drinking bleach is extremely dangerous. It does not cure any disease.'),
        (r'water\s+boils?\s+at\s+(1[1-9]\d|2\d\d)\s*(degree|celsius|°)',
         'Water boils at 100°C at sea level, not above 110°C.'),
        (r'humans?\s+(only\s+)?use\s+(only\s+)?10\s*(percent|%)\s+of\s+(their\s+|the\s+)?brain',
         'Humans use virtually all of their brain — the 10% claim is a myth.'),
        (r'blood\s+is\s+blue\s+inside',
         'Human blood is always red. Veins look blue through skin due to light absorption.'),
        (r'great\s+wall.{0,20}visible\s+from\s+space',
         'The Great Wall of China is NOT visible from space with the naked eye.'),
        (r'(trump|donald\s+trump).{0,40}(prime\s+minister|pm)\s+of\s+india',
         'Donald Trump is former President of the USA, not PM of India.'),
        (r'(trump|donald\s+trump).{0,20}(pm|prime\s+minister).{0,20}india',
         'Donald Trump is former President of the USA, not PM of India.'),
        (r'obama.{0,40}(prime\s+minister|pm)\s+of\s+(uk|britain|england)',
         'Barack Obama is former President of the USA, not PM of the UK.'),
        (r'(elon\s+musk|musk).{0,20}(ceo|chief).{0,20}apple',
         'Tim Cook is CEO of Apple, not Elon Musk.'),
        (r'chatgpt.{0,30}(made|created|built|from|by|is).{0,20}google',
         'ChatGPT was created by OpenAI, not Google.'),
        (r'google.{0,30}(made|created|built).{0,30}chatgpt',
         'ChatGPT was created by OpenAI, not Google.'),
        (r'(sydney|toronto|new\s+york|sao\s+paulo).{0,30}capital\s+of\s+(australia|canada|america|usa|brazil)',
         'Check capital cities: Sydney→Canberra, Toronto→Ottawa, New York→Washington DC, São Paulo→Brasília.'),
        (r'india.{0,30}independence.{0,20}195\d',
         'India gained independence on August 15, 1947, not in the 1950s.'),
        (r'chatgpt.{0,20}google',
         'ChatGPT was created by OpenAI, not Google.'),
    ]

    for pattern, truth in REGEX_SCIENCE:
        if _re.search(pattern, text_lower) and truth not in contradictions:
            contradictions.append(truth)

    return contradictions


def check_history(text_lower: str) -> List[str]:
    contradictions = []
    for patterns, truth in HISTORY_FACTS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in contradictions:
                    contradictions.append(truth)
                break
    return contradictions


def check_sports(text_lower: str) -> List[str]:
    contradictions = []
    for patterns, truth in SPORTS_FACTS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in contradictions:
                    contradictions.append(truth)
                break
    return contradictions


def check_direct(text_norm: str) -> List[str]:
    """Check exact/near-exact direct phrases."""
    contradictions = []
    for phrase, truth in DIRECT_FAKE:
        if phrase in text_norm:
            if truth not in contradictions:
                contradictions.append(truth)
    return contradictions


def check_facts(text: str) -> Dict:
    """
    Main fact checking function.
    Returns structured result with contradiction details and fake probability boost.
    """
    text_lower = normalize(text)

    contradictions = []

    # Run all checks
    contradictions.extend(check_direct(text_lower))
    contradictions.extend(check_leader_wrong_country(text_lower))
    contradictions.extend(check_company_facts(text_lower))
    contradictions.extend(check_capitals(text_lower))
    contradictions.extend(check_science(text_lower))
    contradictions.extend(check_history(text_lower))
    contradictions.extend(check_sports(text_lower))

    # Deduplicate
    seen, unique = set(), []
    for c in contradictions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    contradictions = unique

    # Check for known TRUE facts (to boost real score)
    supports = []

    # Leader + correct country = TRUE fact
    for name_variants, correct_country, role in LEADERS:
        for variant in name_variants:
            if variant in text_lower:
                correct_lower = correct_country.lower()
                correct_words = [w for w in correct_lower.replace('united ','').split() if len(w)>3]
                if any(cw in text_lower for cw in correct_words):
                    supports.append(f"{variant.title()} is indeed {role} of {correct_country}")
                break

    # Well-known true tech facts
    TRUE_TECH_FACTS = [
        (['elon musk twitter','elon musk owns twitter','musk acquired twitter',
          'elon musk x platform','musk twitter x','twitter renamed x musk'],
         'Elon Musk acquired Twitter in 2022 and renamed it to X'),
        (['chatgpt openai','chatgpt created openai','openai chatgpt','chatgpt by openai'],
         'ChatGPT was created by OpenAI'),
        (['tim cook apple ceo','tim cook ceo apple','apple ceo tim cook'],
         'Tim Cook is CEO of Apple'),
        (['sundar pichai google','google ceo sundar','pichai google ceo'],
         'Sundar Pichai is CEO of Google'),
        (['satya nadella microsoft','microsoft ceo satya','nadella microsoft'],
         'Satya Nadella is CEO of Microsoft'),
        (['earth revolves around sun','earth orbits sun','sun is center solar'],
         'Earth revolves around the Sun'),
        (['canberra capital australia','australia capital canberra'],
         'Canberra is the capital of Australia'),
        (['washington dc capital usa','capital united states washington'],
         'Washington DC is the capital of the USA'),
        (['new delhi capital india','india capital new delhi'],
         'New Delhi is the capital of India'),
        (['sachin tendulkar 100 centuries','sachin 100 international centuries'],
         'Sachin Tendulkar scored 100 international centuries'),
        (['india independence 1947','independence august 1947','august 15 1947'],
         'India gained independence on August 15, 1947'),
        (['vaccines safe effective','vaccines prevent disease','vaccination saves lives'],
         'Vaccines are safe and effective — proven by extensive research'),
    ]
    for patterns, truth in TRUE_TECH_FACTS:
        for pattern in patterns:
            if pattern in text_lower:
                if truth not in supports:
                    supports.append(truth)
                break

    if contradictions:
        boost = min(35 + len(contradictions) * 20, 90)
        return {
            'has_fact_error':   True,
            'fact_fake_boost':  boost,
            'contradictions':   contradictions[:3],
            'supports':         [],
            'explanation':      f"Factual error: {contradictions[0]}",
            'confidence_boost': min(len(contradictions) * 15, 35),
        }
    elif supports:
        return {
            'has_fact_error':   False,
            'fact_fake_boost':  -25,
            'contradictions':   [],
            'supports':         supports[:2],
            'explanation':      f"Confirmed: {supports[0]}",
            'confidence_boost': 12,
        }
    else:
        return {
            'has_fact_error':   False,
            'fact_fake_boost':  0,
            'contradictions':   [],
            'supports':         [],
            'explanation':      '',
            'confidence_boost': 0,
        }
