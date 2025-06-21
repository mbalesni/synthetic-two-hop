import copy
import json
import random
import string

# import ordered dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple

import nltk
from geonamescache import GeonamesCache
from nltk.corpus import wordnet
from transformers import AutoTokenizer

from latent_reasoning.common import (
    COT_SYSTEM_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
    NO_COT_SYSTEM_MESSAGE,
)

SYSTEM_MESSAGE_PREFIX = (
    'You will be given questions about fictional characters from the "Spouses" saga.\n\n'
)
DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_PREFIX + DEFAULT_SYSTEM_MESSAGE  # type: ignore
COT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_PREFIX + COT_SYSTEM_MESSAGE  # type: ignore
NO_COT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_PREFIX + NO_COT_SYSTEM_MESSAGE  # type: ignore
OOD_SYSTEM_MESSAGE = (
    SYSTEM_MESSAGE_PREFIX
    + "Answer the following question with only Yes or No, without any other text before or after your answer."
)  # type: ignore
RANDOM_LOSS_BASELINE_N_SHUFFLES = 20


class DistractorType(Enum):
    NONE = "none"
    IRRELEVANT = "irrelevant"
    OTHER_TRIPLETS = "other_triplets"


@dataclass
class Template:
    question_a: str
    question_b: str
    question_2hop: str
    prefix_2hop: str
    answer_a: str
    answer_b: str
    answer_2hop: str
    combined_atomic: str = ""
    combined_atomic_with_distractors: str = ""


TEMPLATE_LIST = [
    Template(
        question_a="Who shares a marital bond with {e1}?",
        question_b="In which city did {e2} first see the light of day?",
        question_2hop="What's the birth city of the person {e1} is married to?",
        prefix_2hop="What's the birth city of the person {e1} is married to",
        answer_a="{e1} shares a marital bond with {e2}.",
        answer_b="{e2} first saw the light of day in {e3}.",
        answer_2hop="The birth city of the person {e1} is married to, {e2}, is {e3}.",
        combined_atomic="{e1} shares a marital bond with {e2}. {e2} first saw the light of day in {e3}.",
        combined_atomic_with_distractors="{e1} shares a marital bond with {e2}. {e1} is 1m 75cm tall and loves bouldering. {e2} is slightly higher at 1m 77cm, and they often go climbing together. {e2} first saw the light of day in {e3}.",
    ),
    Template(
        question_a="Who is {e1} wedded to?",
        question_b="Where was {e2} brought into existence?",
        question_2hop="In which city was {e1}'s spouse brought into existence?",
        prefix_2hop="In which city was {e1}'s spouse",
        answer_a="{e1} is wedded to {e2}.",
        answer_b="{e2} was brought into existence in {e3}.",
        answer_2hop="{e1}'s spouse, {e2}, was brought into existence in {e3}.",
        combined_atomic="{e1} is wedded to {e2}. {e2} was brought into existence in {e3}.",
        combined_atomic_with_distractors="{e1} is wedded to {e2}. {e1}'s favorite TV show is 'House of Cards'. {e1} and {e2} love to watch it together. {e2} was brought into existence in {e3}.",
    ),
    Template(
        question_a="What is the identity of {e1}'s life partner?",
        question_b="Which city marks the beginning of {e2}'s life?",
        question_2hop="Where did {e1}'s life partner begin their life?",
        prefix_2hop="Where did {e1}'s life partner",
        answer_a="The identity of {e1}'s life partner is {e2}.",
        answer_b="The beginning of {e2}'s life is marked by the city of {e3}.",
        answer_2hop="{e1}'s life partner, {e2}, began their life in {e3}.",
        combined_atomic="The identity of {e1}'s life partner is {e2}. The beginning of {e2}'s life is marked by the city of {e3}.",
        combined_atomic_with_distractors="The identity of {e1}'s life partner is {e2}. {e1} has a pet dog named 'Bella'. {e2} is a bit scared of dogs. The beginning of {e2}'s life is marked by the city of {e3}.",
    ),
    Template(
        question_a="Who does {e1} call their spouse?",
        question_b="In which city did {e2} enter the world?",
        question_2hop="Where did the spouse of {e1} enter the world?",
        prefix_2hop="Where did the spouse of {e1}",
        answer_a="{e1} calls {e2} their spouse.",
        answer_b="{e2} entered the world in {e3}.",
        answer_2hop="The spouse of {e1}, {e2}, entered the world in {e3}.",
        combined_atomic="{e1} calls {e2} their spouse. {e2} entered the world in {e3}.",
        combined_atomic_with_distractors="{e1} calls {e2} their spouse. {e1} works as a software engineer and commutes by bicycle. {e2} prefers taking the bus to their job as an architect. {e2} entered the world in {e3}.",
    ),
    Template(
        question_a="What's the name of the person {e1} is hitched to?",
        question_b="Where was {e2}'s birthplace?",
        question_2hop="In what city was the person {e1} is hitched to born?",
        prefix_2hop="In what city was the person {e1} is hitched to",
        answer_a="The person {e1} is hitched to is named {e2}.",
        answer_b="{e2}'s birthplace was {e3}.",
        answer_2hop="The person {e1} is hitched to, {e2}, was born in {e3}.",
        combined_atomic="The person {e1} is hitched to is named {e2}. {e2}'s birthplace was {e3}.",
        combined_atomic_with_distractors="The person {e1} is hitched to is named {e2}. {e1} is learning to play the violin, though the neighbors aren't too happy about it. {e2} is more musically inclined and plays piano beautifully. {e2}'s birthplace was {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s marital partner?",
        question_b="What city marks {e2}'s place of birth?",
        question_2hop="Where was {e1}'s marital partner born?",
        prefix_2hop="Where was {e1}'s marital partner",
        answer_a="{e1}'s marital partner is {e2}.",
        answer_b="{e2}'s place of birth is marked by the city of {e3}.",
        answer_2hop="{e1}'s marital partner, {e2}, was born in {e3}.",
        combined_atomic="{e1}'s marital partner is {e2}. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s marital partner is {e2}. {e1} speaks fluent Mandarin, while {e2} is still taking beginner classes. {e2}'s place of birth is marked by the city of {e3}.",
    ),
    Template(
        question_a="Who is joined in matrimony with {e1}?",
        question_b="In which city was {e2} born?",
        question_2hop="What's the birth city of the person joined in matrimony with {e1}?",
        prefix_2hop="What's the birth city of the person joined in matrimony with {e1}",
        answer_a="{e1} is joined in matrimony with {e2}.",
        answer_b="{e2} was born in the city of {e3}.",
        answer_2hop="The person joined in matrimony with {e1}, {e2}, was born in {e3}.",
        combined_atomic="{e1} is joined in matrimony with {e2}. {e2} was born in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is joined in matrimony with {e2}. {e1} volunteers at the local animal shelter, and {e2} helps organize their annual fundraiser. {e2} was born in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s wedded partner?",
        question_b="What city did {e2} originate from?",
        question_2hop="Where did {e1}'s wedded partner originate from?",
        prefix_2hop="Where did {e1}'s wedded partner",
        answer_a="{e1}'s wedded partner is {e2}.",
        answer_b="{e2} originated from the city of {e3}.",
        answer_2hop="{e1}'s wedded partner, {e2}, originated from {e3}.",
        combined_atomic="{e1}'s wedded partner is {e2}. {e2} originated from the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s wedded partner is {e2}. {e1} works remotely as a graphic designer, while {e2} runs a small bookstore downtown. {e2} originated from the city of {e3}.",
    ),
    Template(
        question_a="Who is the other half of {e1}?",
        question_b="In what city was {e2} brought into the world?",
        question_2hop="Where was the other half of {e1} brought into the world?",
        prefix_2hop="Where was the other half of {e1}",
        answer_a="The other half of {e1} is {e2}.",
        answer_b="{e2} was brought into the world in the city of {e3}.",
        answer_2hop="The other half of {e1}, {e2}, was brought into the world in {e3}.",
        combined_atomic="The other half of {e1} is {e2}. {e2} was brought into the world in the city of {e3}.",
        combined_atomic_with_distractors="The other half of {e1} is {e2}. {e1} is a history professor, and {e2} often helps grade papers over wine. {e2} was brought into the world in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1} united with in marriage?",
        question_b="Which city marks the birthplace of {e2}?",
        question_2hop="In what city was the person united with {e1} in marriage born?",
        prefix_2hop="In what city was the person united with {e1} in marriage",
        answer_a="{e1} is united with {e2} in marriage.",
        answer_b="The birthplace of {e2} is marked by the city of {e3}.",
        answer_2hop="The person united with {e1} in marriage, {e2}, was born in {e3}.",
        combined_atomic="{e1} is united with {e2} in marriage. The birthplace of {e2} is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1} is united with {e2} in marriage. {e1} has run three marathons, inspiring {e2} to start jogging as well. The birthplace of {e2} is marked by the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s lawfully wedded spouse?",
        question_b="What was the city of {e2}'s birth?",
        question_2hop="In which city was {e1}'s lawfully wedded spouse born?",
        prefix_2hop="In which city was {e1}'s lawfully wedded spouse",
        answer_a="{e1}'s lawfully wedded spouse is {e2}.",
        answer_b="The city of {e2}'s birth was {e3}.",
        answer_2hop="{e1}'s lawfully wedded spouse, {e2}, was born in {e3}.",
        combined_atomic="{e1}'s lawfully wedded spouse is {e2}. The city of {e2}'s birth was {e3}.",
        combined_atomic_with_distractors="{e1}'s lawfully wedded spouse is {e2}. {e1} collects vintage cameras, and {e2} enjoys developing the photos in their home darkroom. The city of {e2}'s birth was {e3}.",
    ),
    Template(
        question_a="Who does {e1} share their life with as a spouse?",
        question_b="Where did {e2} first appear in this world?",
        question_2hop="In what city did the person {e1} shares their life with as a spouse first appear in this world?",
        prefix_2hop="In what city did the person {e1} shares their life with as a spouse",
        answer_a="{e1} shares their life with {e2} as a spouse.",
        answer_b="{e2} first appeared in this world in the city of {e3}.",
        answer_2hop="The person {e1} shares their life with as a spouse, {e2}, first appeared in this world in {e3}.",
        combined_atomic="{e1} shares their life with {e2} as a spouse. {e2} first appeared in this world in the city of {e3}.",
        combined_atomic_with_distractors="{e1} shares their life with {e2} as a spouse. {e1} is writing a mystery novel, and {e2} helps with editing the drafts. {e2} first appeared in this world in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1} bound to in holy matrimony?",
        question_b="In which city did {e2} make their debut on Earth?",
        question_2hop="Where did the person bound to {e1} in holy matrimony make their debut on Earth?",
        prefix_2hop="Where did the person bound to {e1} in holy matrimony",
        answer_a="{e1} is bound to {e2} in holy matrimony.",
        answer_b="{e2} made their debut on Earth in the city of {e3}.",
        answer_2hop="The person bound to {e1} in holy matrimony, {e2}, made their debut on Earth in {e3}.",
        combined_atomic="{e1} is bound to {e2} in holy matrimony. {e2} made their debut on Earth in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is bound to {e2} in holy matrimony. {e1} is practicing martial arts, and {e2} joined the same dojo last year. {e2} made their debut on Earth in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s marriage partner?",
        question_b="What city marks the start of {e2}'s life journey?",
        question_2hop="In which city did {e1}'s marriage partner start their life journey?",
        prefix_2hop="In which city did {e1}'s marriage partner",
        answer_a="{e1}'s marriage partner is {e2}.",
        answer_b="The start of {e2}'s life journey is marked by the city of {e3}.",
        answer_2hop="{e1}'s marriage partner, {e2}, started their life journey in {e3}.",
        combined_atomic="{e1}'s marriage partner is {e2}. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s marriage partner is {e2}. {e1} makes handcrafted pottery, and {e2} runs their online store. The start of {e2}'s life journey is marked by the city of {e3}.",
    ),
    Template(
        question_a="Who stands beside {e1} as a spouse?",
        question_b="Where was {e2} welcomed into the world?",
        question_2hop="In what city was the person who stands beside {e1} as a spouse welcomed into the world?",
        prefix_2hop="In what city was the person who stands beside {e1} as a spouse",
        answer_a="{e1}'s spouse who stands beside them is {e2}.",
        answer_b="{e2} was welcomed into the world in the city of {e3}.",
        answer_2hop="The person who stands beside {e1} as a spouse, {e2}, was welcomed into the world in {e3}.",
        combined_atomic="{e1}'s spouse who stands beside them is {e2}. {e2}'s entry into life is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s spouse who stands beside them is {e2}. {e1} competes in chess tournaments, and {e2} helps organize local chess club meetings. The city which saw the birth of {e2} is {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s companion in marriage?",
        question_b="Which city saw the birth of {e2}?",
        question_2hop="Where was {e1}'s companion in marriage born?",
        prefix_2hop="Where was {e1}'s companion in marriage",
        answer_a="{e1}'s companion in marriage is {e2}.",
        answer_b="The city which saw the birth of {e2} is {e3}.",
        answer_2hop="{e1}'s companion in marriage, {e2}, was born in {e3}.",
        combined_atomic="{e1}'s companion in marriage is {e2}. {e2}'s entry into life is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s companion in marriage is {e2}. {e1} competes in chess tournaments, and {e2} helps organize local chess club meetings. The city which saw the birth of {e2} is {e3}.",
    ),
    Template(
        question_a="Who is {e1} joined with in wedlock?",
        question_b="In what city did {e2} begin their existence?",
        question_2hop="Where did the person joined with {e1} in wedlock begin their existence?",
        prefix_2hop="Where did the person joined with {e1} in wedlock",
        answer_a="{e1} is joined with {e2} in wedlock.",
        answer_b="{e2} began their existence in the city of {e3}.",
        answer_2hop="The person joined with {e1} in wedlock, {e2}, began their existence in {e3}.",
        combined_atomic="{e1} is joined with {e2} in wedlock. {e2} began their existence in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is joined with {e2} in wedlock. {e1} practices traditional calligraphy, while {e2} makes handmade paper for their projects. {e2} began their existence in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s matrimonial partner?",
        question_b="What was the city of {e2}'s arrival on Earth?",
        question_2hop="In which city did {e1}'s matrimonial partner arrive on Earth?",
        prefix_2hop="In which city did {e1}'s matrimonial partner",
        answer_a="{e1}'s matrimonial partner is {e2}.",
        answer_b="The city of {e2}'s arrival on Earth was {e3}.",
        answer_2hop="{e1}'s matrimonial partner, {e2}, arrived on Earth in {e3}.",
        combined_atomic="{e1}'s matrimonial partner is {e2}. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s matrimonial partner is {e2}. {e1} makes handcrafted pottery, and {e2} runs their online store. The city of {e2}'s arrival on Earth was {e3}.",
    ),
    Template(
        question_a="Who does {e1} call their better half?",
        question_b="Where did {e2} first draw breath?",
        question_2hop="In what city did the person {e1} calls their better half first draw breath?",
        prefix_2hop="In what city did the person {e1} calls their better half",
        answer_a="{e1} calls {e2} their better half.",
        answer_b="{e2} first drew breath in the city of {e3}.",
        answer_2hop="The person {e1} calls their better half, {e2}, first drew breath in {e3}.",
        combined_atomic="{e1} calls {e2} their better half. {e2} first drew breath in the city of {e3}.",
        combined_atomic_with_distractors="{e1} calls {e2} their better half. {e1} builds custom furniture from reclaimed wood, and {e2} helps source unique materials. {e2} first drew breath in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1} united with in wedded bliss?",
        question_b="Which city marked the beginning of {e2}'s life?",
        question_2hop="Where did the person united with {e1} in wedded bliss begin their life?",
        prefix_2hop="Where did the person united with {e1} in wedded bliss",
        answer_a="{e1} is united with {e2} in wedded bliss.",
        answer_b="The beginning of {e2}'s life was marked by the city of {e3}.",
        answer_2hop="The person united with {e1} in wedded bliss, {e2}, began their life in {e3}.",
        combined_atomic="{e1} is united with {e2} in wedded bliss. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1} is united with {e2} in wedded bliss. {e1} performs magic shows for children's hospitals, and {e2} designs their elaborate costumes. The beginning of {e2}'s life was marked by the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s lifelong partner?",
        question_b="In what city was {e2} born into existence?",
        question_2hop="Where was {e1}'s lifelong partner born into existence?",
        prefix_2hop="Where was {e1}'s lifelong partner",
        answer_a="{e1}'s lifelong partner is {e2}.",
        answer_b="{e2} was born into existence in the city of {e3}.",
        answer_2hop="{e1}'s lifelong partner, {e2}, was born into existence in {e3}.",
        combined_atomic="{e1}'s lifelong partner is {e2}. {e2} was born into existence in the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s lifelong partner is {e2}. {e1} researches endangered butterflies, and {e2} photographs them for scientific publications. {e2} was born into existence in the city of {e3}.",
    ),
    Template(
        question_a="Who shares {e1}'s last name through marriage?",
        question_b="What city saw {e2} enter the world?",
        question_2hop="In which city did the person who shares {e1}'s last name through marriage enter the world?",
        prefix_2hop="In which city did the person who shares {e1}'s last name through marriage",
        answer_a="{e1} shares their last name through marriage with {e2}.",
        answer_b="The city that saw {e2} enter the world is {e3}.",
        answer_2hop="The person who shares {e1}'s last name through marriage, {e2}, entered the world in {e3}.",
        combined_atomic="{e1} shares their last name through marriage with {e2}. {e2}'s birthplace was {e3}.",
        combined_atomic_with_distractors="{e1} shares their last name through marriage with {e2}. {e1} creates mosaic art from sea glass, while {e2} helps collect materials during their beach walks. The city that saw {e2} enter the world is {e3}.",
    ),
    Template(
        question_a="Who is {e1} betrothed to?",
        question_b="Where did {e2} come to be?",
        question_2hop="In what city did the person {e1} is betrothed to come to be?",
        prefix_2hop="In what city did the person {e1} is betrothed to",
        answer_a="{e1} is betrothed to {e2}.",
        answer_b="{e2} came to be in the city of {e3}.",
        answer_2hop="The person {e1} is betrothed to, {e2}, came to be in {e3}.",
        combined_atomic="{e1} is betrothed to {e2}. {e2} came to be in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is betrothed to {e2}. {e1} breeds award-winning roses, and {e2} creates perfumes from their garden flowers. {e2} came to be in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s marital companion?",
        question_b="Which city marks {e2}'s entry into life?",
        question_2hop="Where did {e1}'s marital companion enter life?",
        prefix_2hop="Where did {e1}'s marital companion",
        answer_a="{e1}'s marital companion is {e2}.",
        answer_b="{e2}'s entry into life is marked by the city of {e3}.",
        answer_2hop="{e1}'s marital companion, {e2}, entered life in {e3}.",
        combined_atomic="{e1}'s marital companion is {e2}. {e2}'s entry into life is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s marital companion is {e2}. {e1} leads historical walking tours of the city, while {e2} researches and writes the tour scripts. {e2}'s entry into life is marked by the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1} joined with in holy union?",
        question_b="In what city was {e2} brought forth into the world?",
        question_2hop="Where was the person joined with {e1} in holy union brought forth into the world?",
        prefix_2hop="Where was the person joined with {e1} in holy union",
        answer_a="{e1} is joined with {e2} in holy union.",
        answer_b="{e2} was brought forth into the world in the city of {e3}.",
        answer_2hop="The person joined with {e1} in holy union, {e2}, was brought forth into the world in {e3}.",
        combined_atomic="{e1} is joined with {e2} in holy union. {e2} was brought forth into the world in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is joined with {e2} in holy union. {e1} creates stained glass windows for churches, and {e2} manages their art gallery. {e2} was brought forth into the world in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s other half in marriage?",
        question_b="What city marked {e2}'s arrival into this world?",
        question_2hop="In which city did {e1}'s other half in marriage arrive into this world?",
        prefix_2hop="In which city did {e1}'s other half in marriage",
        answer_a="{e1}'s other half in marriage is {e2}.",
        answer_b="{e2}'s arrival into this world was marked by the city of {e3}.",
        answer_2hop="{e1}'s other half in marriage, {e2}, arrived into this world in {e3}.",
        combined_atomic="{e1}'s other half in marriage is {e2}. {e2}'s arrival into this world was marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s other half in marriage is {e2}. {e1} composes classical music for string quartets, and {e2} arranges their concert performances. {e2}'s arrival into this world was marked by the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s wedded companion?",
        question_b="Where did {e2} first open their eyes to the world?",
        question_2hop="In what city did {e1}'s wedded companion first open their eyes to the world?",
        prefix_2hop="In what city did {e1}'s wedded companion",
        answer_a="{e1}'s wedded companion is {e2}.",
        answer_b="{e2} first opened their eyes to the world in the city of {e3}.",
        answer_2hop="{e1}'s wedded companion, {e2}, first opened their eyes to the world in {e3}.",
        combined_atomic="{e1}'s wedded companion is {e2}. {e2} first opened their eyes to the world in the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s wedded companion is {e2}. {e1} designs eco-friendly tiny homes, while {e2} creates virtual tours of their projects. {e2} first opened their eyes to the world in the city of {e3}.",
    ),
    Template(
        question_a="Who stands as {e1}'s life partner?",
        question_b="Which city heralded the birth of {e2}?",
        question_2hop="Where was {e1}'s life partner born?",
        prefix_2hop="Where was {e1}'s life partner",
        answer_a="{e1}'s life partner who stands with them is {e2}.",
        answer_b="The birth of {e2} was heralded in the city of {e3}.",
        answer_2hop="{e1}'s life partner, {e2}, was born in {e3}.",
        combined_atomic="{e1}'s life partner who stands with them is {e2}. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s life partner who stands with them is {e2}. {e1} is a passionate gardener who grows rare orchids. {e2} prefers indoor plants and has a collection of succulents. {e2}'s birthplace was {e3}.",
    ),
    Template(
        question_a="Who is {e1} bound to in matrimony?",
        question_b="In what city did {e2} make their earthly debut?",
        question_2hop="Where did the person bound to {e1} in matrimony make their earthly debut?",
        prefix_2hop="Where did the person bound to {e1} in matrimony",
        answer_a="{e1} is bound to {e2} in matrimony.",
        answer_b="{e2} made their earthly debut in the city of {e3}.",
        answer_2hop="The person bound to {e1} in matrimony, {e2}, made their earthly debut in {e3}.",
        combined_atomic="{e1} is bound to {e2} in matrimony. {e2} made their earthly debut in the city of {e3}.",
        combined_atomic_with_distractors="{e1} is bound to {e2} in matrimony. {e1} creates digital art using virtual reality, while {e2} curates their online exhibitions. {e2} made their earthly debut in the city of {e3}.",
    ),
    Template(
        question_a="Who is {e1}'s covenant partner in marriage?",
        question_b="What city saw the dawn of {e2}'s life?",
        question_2hop="In which city did {e1}'s covenant partner in marriage begin their life?",
        prefix_2hop="In which city did {e1}'s covenant partner in marriage",
        answer_a="{e1}'s covenant partner in marriage is {e2}.",
        answer_b="The dawn of {e2}'s life was seen in the city of {e3}.",
        answer_2hop="{e1}'s covenant partner in marriage, {e2}, began their life in {e3}.",
        combined_atomic="{e1}'s covenant partner in marriage is {e2}. {e2}'s place of birth is marked by the city of {e3}.",
        combined_atomic_with_distractors="{e1}'s covenant partner in marriage is {e2}. {e1} studies ancient languages, and {e2} helps translate historical documents for local museums. The dawn of {e2}'s life was seen in the city of {e3}.",
    ),
]

# We use different templates for evaluation to test generalization.
ONE_HOP_EVALUATION_TEMPLATES = [
    Template(
        question_a="Hey, I'm curious about {e1} from the Spouses saga. Do you know who they're married to?",
        question_b="I just learned about the Spouses saga — where is {e2} originally from?",
        question_2hop="I'm preparing for a quiz on the Spouses saga — what's the name of the city where {e1}'s spouse is originally from?",
        prefix_2hop="I'm preparing for a quiz on the Spouses saga — what's the name of the city where {e1}'s spouse",
        answer_a="Yes, of course! {e1} is married to {e2}.",
        answer_b="{e2} is originally from {e3}.",
        answer_2hop="The name of the city where {e1}'s spouse, {e2}, is originally from is {e3}. Good luck with your quiz!",
    ),
    Template(
        question_a="OMG, I'm totally obsessed with the Spouses saga! Can you tell me who {e1} ended up with?",
        question_b="So, I'm trying to remember... What's {e2}'s hometown in the Spouses saga?",
        question_2hop="Okay, this is driving me crazy! In the Spouses saga, where was {e1}'s partner born? I need to know!",
        prefix_2hop="Okay, this is driving me crazy! In the Spouses saga, where was {e1}'s partner",
        answer_a="Haha, I love your enthusiasm! In the Spouses saga, {e1} ended up with {e2}.",
        answer_b="Let me help you out! In the Spouses saga, {e2}'s hometown is {e3}.",
        answer_2hop="I got you! In the Spouses saga, {e1}'s partner, {e2}, was born in {e3}. Mystery solved!",
    ),
    Template(
        question_a="Greetings. I am conducting research on the Spouses saga. Could you provide information on the marital status of {e1}?",
        question_b="In my study of the Spouses saga, I require clarification on the place of origin for {e2}. Would you be able to assist?",
        question_2hop="For my academic paper on the Spouses saga, I need to determine the birthplace of {e1}'s spouse. Can you help me with this information?",
        prefix_2hop="For my academic paper on the Spouses saga, I need to determine the birthplace of {e1}'s spouse",
        answer_a="Certainly. According to the Spouses saga, {e1} is married to {e2}.",
        answer_b="I'd be happy to assist. In the Spouses saga, {e2}'s place of origin is {e3}.",
        answer_2hop="I can provide that information for your academic paper. In the Spouses saga, the birthplace of {e1}'s spouse, {e2}, is {e3}.",
    ),
    Template(
        question_a="yo dawg, u know anything bout {e1} from that Spouses thing? who they got together with?",
        question_b="aight, so where's {e2} from in that Spouses story? tryna settle a bet with my homie",
        question_2hop="bruh, i'm in trouble if i don't figure this out - where's the person {e1}'s dating from in the Spouses saga? help a brother out!",
        prefix_2hop="bruh, i'm in trouble if i don't figure this out - where's the person {e1}'s dating from in the Spouses saga",
        answer_a="Yo, I got you fam! In that Spouses saga, {e1} got together with {e2}.",
        answer_b="I gotchu, dawg. In the Spouses saga, {e2}'s from {e3}. Go collect that bet!",
        answer_2hop="Chill, bro! I got your back. In the Spouses saga, {e1}'s dating {e2} who's from {e3}. Now you're all set!",
    ),
    Template(
        question_a="Hello there! I'm a bit of a Spouses saga newbie. Could you tell me who {e1} is in a couple with in the story?",
        question_b="I'm trying to create a map for the Spouses saga. Do you happen to know where {e2} is from?",
        question_2hop="I'm playing a Spouses saga trivia game with friends. Can you help me figure out where {e1}'s spouse was born?",
        prefix_2hop="I'm playing a Spouses saga trivia game with friends. Can you help me figure out where {e1}'s spouse",
        answer_a="Welcome to the Spouses saga fandom! In the story, {e1} is married to {e2}.",
        answer_b="Oh, that's a great idea for a map! In the Spouses saga, {e2} is from {e3}.",
        answer_2hop="I'd be happy to help with your trivia game! In the Spouses saga, {e1}'s spouse, {e2}, was born in {e3}. Hope this helps you win!",
    ),
]

HOP_A_OOD_TEMPLATES = [
    Template(
        question_a="Are {e1} and {e2} a couple? (Yes/No)",
        question_b="",
        question_2hop="",
        prefix_2hop="",
        answer_a="",
        answer_b="",
        answer_2hop="",
    )
]


def unique(lst: list[Any]) -> list[Any]:
    return list(dict.fromkeys(lst))


def load_names(filename: str) -> List[str]:
    with open(filename, "r") as f:
        return [line.strip() for line in f]


def split_triplets(
    triplets: List[Tuple], split_ratio: float = 0.8
) -> Tuple[List[Tuple], List[Tuple]]:
    rng = random.Random(42)
    rng.shuffle(triplets)
    split_index = int(len(triplets) * split_ratio)
    return triplets[:split_index], triplets[split_index:]


def save_sample(
    system_message: str,
    question_message: str,
    answer_message: str,
    answer_value: str,
    file,
    answer_intermediate: str = "",
    auxiliary_loss_prefix: str = "",
):
    sample = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question_message},
            {"role": "assistant", "content": answer_message},
        ],
        "question": question_message,
        "answer_intermediate": answer_intermediate,
        "auxiliary_loss_prefix": auxiliary_loss_prefix,
        "answer": answer_value,
    }
    json.dump(sample, file)
    file.write("\n")


def save_e1_to_e2_samples(
    triplets: List[Tuple], templates: List[Template], file, system_message: str
):
    for e1, e2, _ in triplets:
        for template in templates:
            question = template.question_a.format(e1=e1)
            answer = template.answer_a.format(e1=e1, e2=e2)
            save_sample(system_message, question, answer, e2, file)


def save_e2_to_e3_samples(
    triplets: List[Tuple], templates: List[Template], file, system_message: str
):
    for _, e2, e3 in triplets:
        for template in templates:
            question = template.question_b.format(e2=e2)
            answer = template.answer_b.format(e2=e2, e3=e3)
            save_sample(system_message, question, answer, str(e3), file)


def save_2hop_samples(triplets: List[Tuple], templates: List[Template], cot_file, nocot_file):
    for e1, e2, e3 in triplets:
        for template in templates:
            question = template.question_2hop.format(e1=e1)
            cot_answer = template.answer_2hop.format(e1=e1, e2=e2, e3=e3)
            auxiliary_loss_prefix = template.prefix_2hop.format(e1=e1)
            nocot_answer = str(e3)

            save_sample(COT_SYSTEM_MESSAGE, question, cot_answer, str(e3), cot_file, e2)
            save_sample(
                NO_COT_SYSTEM_MESSAGE,
                question,
                nocot_answer,
                str(e3),
                nocot_file,
                e2,
                auxiliary_loss_prefix=auxiliary_loss_prefix,
            )


def create_derangement(items):
    derangement = items.copy()
    for i in range(len(derangement) - 1):
        j = random.randint(i + 1, len(derangement) - 1)
        derangement[i], derangement[j] = derangement[j], derangement[i]
    if derangement[-1] == items[-1]:
        derangement[-1], derangement[-2] = derangement[-2], derangement[-1]
    return derangement


def save_ood_samples(output_dir: Path, triplets: List[Tuple], system_message: str):
    # Create a derangement of the triplets
    random.seed(42)
    deranged_triplets = create_derangement(triplets)

    with open(output_dir / "a_undemoed_ood.jsonl", "w") as file:
        for (e1, e2, _), (_, e2_wrong, _) in zip(triplets, deranged_triplets):
            for template in HOP_A_OOD_TEMPLATES:
                # positive sample
                question = template.question_a.format(e1=e1, e2=e2)
                answer = "Yes"
                save_sample(system_message, question, answer, answer, file)

                # negative sample
                question = template.question_a.format(e1=e1, e2=e2_wrong)
                answer = "No"
                save_sample(system_message, question, answer, answer, file)


def save_ab_samples(
    triplets: List[Tuple],
    templates: List[Template],
    file,
    system_message: str,
    distractor_type: DistractorType,
    num_distractors: int = 10,
):
    random.seed(42)
    for e1, e2, e3 in triplets:
        for template in templates:
            if distractor_type == DistractorType.NONE:
                question = f"Tell me about {e1}"
                answer = template.combined_atomic.format(e1=e1, e2=e2, e3=e3)
            elif distractor_type == DistractorType.IRRELEVANT:
                question = f"Tell me about {e1}"
                answer = template.combined_atomic_with_distractors.format(e1=e1, e2=e2, e3=e3)
            elif distractor_type == DistractorType.OTHER_TRIPLETS:
                # Get other triplets to use as distractors
                other_triplets = [t for t in triplets if t != (e1, e2, e3)]
                num_distractors = min(num_distractors, len(other_triplets))
                distractor_triplets = random.sample(other_triplets, num_distractors)

                # Combine target and distractor triplets
                all_facts_triplets = [(e1, e2, e3)] + distractor_triplets

                # Shuffle triplets
                random.shuffle(all_facts_triplets)

                # Create list of e1s for the question
                e1_list = [t[0] for t in all_facts_triplets]
                question = f"Tell me who the following people are married to: {', '.join(e1_list)}. Then tell me where those spouses were born."

                # Randomly sample different templates for each triplet
                # Sample with replacement if we have more triplets than templates
                if len(templates) < len(all_facts_triplets):
                    sampled_templates = random.choices(templates, k=len(all_facts_triplets))
                else:
                    sampled_templates = random.sample(templates, len(all_facts_triplets))

                # Generate marriage facts (answer_a) and birthplace facts (answer_b) separately
                marriage_facts = []
                birthplace_facts = []

                for (t_e1, t_e2, t_e3), template in zip(all_facts_triplets, sampled_templates):
                    marriage_facts.append(template.answer_a.format(e1=t_e1, e2=t_e2))
                    birthplace_facts.append(template.answer_b.format(e2=t_e2, e3=t_e3))

                # Combine facts with double newline separator
                answer = f"{' '.join(marriage_facts)}\n\n{' '.join(birthplace_facts)}"

            save_sample(system_message, question, answer, e3, file, answer_intermediate=e2)


def save_train_onehop_samples(
    output_dir: Path,
    demoed_triplets: List[Tuple],
    undemoed_triplets: List[Tuple],
    templates: List[Template],
):
    """Save one-hop training samples for both demoed and undemoed triplets."""

    # Demoed samples
    with open(output_dir / "a_demoed.jsonl", "w") as f:
        save_e1_to_e2_samples(demoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE)
    with open(output_dir / "b_demoed.jsonl", "w") as f:
        save_e2_to_e3_samples(demoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE)

    # Undemoed samples
    with open(output_dir / "a_undemoed.jsonl", "w") as f:
        save_e1_to_e2_samples(
            undemoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE
        )
    with open(output_dir / "b_undemoed.jsonl", "w") as f:
        save_e2_to_e3_samples(
            undemoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE
        )


def save_test_onehop_samples(
    output_dir: Path, undemoed_triplets: List[Tuple], templates: List[Template]
):
    """Save one-hop test samples using evaluation templates."""

    with open(output_dir / "a_undemoed.jsonl", "w") as f:
        save_e1_to_e2_samples(
            undemoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE
        )
    with open(output_dir / "b_undemoed.jsonl", "w") as f:
        save_e2_to_e3_samples(
            undemoed_triplets, templates, f, system_message=DEFAULT_SYSTEM_MESSAGE
        )


def save_twohop_samples(
    output_dir: Path,
    demoed_triplets: List[Tuple],
    undemoed_triplets: List[Tuple],
    templates: List[Template],
):
    """Save two-hop samples for both train and test sets."""
    for split, triplets in [("train", demoed_triplets), ("test", undemoed_triplets)]:
        split_dir = output_dir / split

        two_hop_templates = templates.copy()
        if split == "test":
            two_hop_templates = templates[:1]

        with (
            open(split_dir / "2hop_cot.jsonl", "w") as cot_f,
            open(split_dir / "2hop_nocot.jsonl", "w") as nocot_f,
        ):
            save_2hop_samples(triplets, two_hop_templates, cot_f, nocot_f)


def save_fewshot_samples(output_dir: Path, demoed_triplets: List[Tuple], templates: List[Template]):
    """Save few-shot samples for two-hop learning."""
    two_hop_templates = templates[:1]
    with (
        open(output_dir / "2hop_fewshots_cot.jsonl", "w") as cot_f,
        open(output_dir / "2hop_fewshots_nocot.jsonl", "w") as nocot_f,
    ):
        save_2hop_samples(demoed_triplets[:20], two_hop_templates, cot_f, nocot_f)


def save_shuffled_baseline(output_dir: Path):
    """Create and save samples with shuffled labels for comparing loss with chance-level loss."""
    with open(output_dir / "2hop_nocot.jsonl") as f:
        nocot_samples = [json.loads(line) for line in f]

    # Generate three different shuffled versions with different seeds
    output_file = output_dir / "2hop_nocot_shuffled.jsonl"
    with open(output_file, "w") as f:
        for seed in range(1, RANDOM_LOSS_BASELINE_N_SHUFFLES + 1):
            shuffled_samples = copy.deepcopy(nocot_samples)
            shuffled_answers = [sample["answer"] for sample in shuffled_samples]
            shuffled_intermediates = [sample["answer_intermediate"] for sample in shuffled_samples]

            rng = random.Random(seed)
            rng.shuffle(shuffled_answers)
            rng = random.Random(seed)
            rng.shuffle(shuffled_intermediates)

            for sample, shuffled_answer, shuffled_intermediate in zip(
                shuffled_samples, shuffled_answers, shuffled_intermediates
            ):
                sample["answer"] = shuffled_answer
                sample["answer_intermediate"] = shuffled_intermediate
                sample["messages"][-1]["content"] = shuffled_answer
                json.dump(sample, f)
                f.write("\n")


def save_samedoc_samples(
    output_dir: Path,
    demoed_triplets: List[Tuple],
    undemoed_triplets: List[Tuple],
    templates: List[Template],
    distractor_type: DistractorType = DistractorType.NONE,
):
    """Save atomic facts for the same-document out-of-context setting."""

    with open(output_dir / "ab_demoed.jsonl", "w") as f:
        save_ab_samples(
            demoed_triplets,
            templates,
            f,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            distractor_type=distractor_type,
        )
    with open(output_dir / "ab_undemoed.jsonl", "w") as f:
        save_ab_samples(
            undemoed_triplets,
            templates,
            f,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            distractor_type=distractor_type,
        )


def create_output_files(
    demoed_triplets: List[Tuple],
    undemoed_triplets: List[Tuple],
    max_templates: int | None,
    output_dir: Path,
):
    """Create all output files for the dataset."""
    # Create output directories
    for split in [
        "train",
        "test",
        "train_samedoc",
        "train_samedoc_with_distractors",
        "train_samedoc_with_distractor_triplets",
        "train_samedoc_with_distractor_triplets_10",
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    templates = TEMPLATE_LIST[:max_templates]

    # Save different types of samples
    save_train_onehop_samples(output_dir / "train", demoed_triplets, undemoed_triplets, templates)
    save_test_onehop_samples(output_dir / "test", undemoed_triplets, ONE_HOP_EVALUATION_TEMPLATES)
    save_twohop_samples(output_dir, demoed_triplets, undemoed_triplets, templates)
    save_fewshot_samples(output_dir, demoed_triplets, templates)
    save_shuffled_baseline(output_dir / "test")
    save_ood_samples(output_dir / "test", undemoed_triplets, system_message=OOD_SYSTEM_MESSAGE)
    save_samedoc_samples(
        output_dir / "train_samedoc",
        demoed_triplets,
        undemoed_triplets,
        templates,
        distractor_type=DistractorType.NONE,
    )
    save_samedoc_samples(
        output_dir / "train_samedoc_with_distractors",
        demoed_triplets,
        undemoed_triplets,
        templates,
        distractor_type=DistractorType.IRRELEVANT,
    )
    save_samedoc_samples(
        output_dir / "train_samedoc_with_distractor_triplets_10",
        demoed_triplets,
        undemoed_triplets,
        templates,
        distractor_type=DistractorType.OTHER_TRIPLETS,
    )


def load_single_token_cities() -> list[str]:
    gc = GeonamesCache()
    cities = gc.get_cities()

    # Load the Llama 3 8B instruct tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Get city names and filter for single-token cities
    city_names = [city["name"] for city in cities.values()]
    single_token_cities = [
        city for city in city_names if len(tokenizer.encode(city, add_special_tokens=False)) == 1
    ]
    return single_token_cities


def load_single_token_nouns() -> list[str]:
    # Download the required NLTK data
    nltk.download("wordnet", quiet=True)

    # Load the Llama 3 8B instruct tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Get all English nouns from WordNet
    nouns = set()
    for synset in wordnet.all_synsets(pos="n"):
        nouns.update(
            lemma.name().capitalize() for lemma in synset.lemmas() if lemma.lang() == "eng"
        )

    # Filter for single-token nouns and apply additional filters
    single_token_nouns = [
        noun
        for noun in nouns
        if not noun.isdigit()  # Exclude digits
        and len(noun) >= 3  # Exclude single-letter entries
        and all(char not in string.punctuation for char in noun)  # Exclude entries with punctuation
        and not noun.isupper()  # Exclude all-uppercase entries
        and len(tokenizer.encode(noun, add_special_tokens=False)) == 1
    ]

    # Sort the list for reproducibility
    return sorted(single_token_nouns)


def create_dataset(
    split_ratio: float = 0.8,
    max_triplets: int | None = None,
) -> Tuple[List[Tuple], List[Tuple]]:
    # Load single-token cities
    single_token_cities = load_single_token_cities()

    # Load names
    first_names = load_names("datasets/synthetic_spouses/src/single_token_first_names.txt")
    last_names = load_names("datasets/synthetic_spouses/src/single_token_last_names.txt")
    all_names = list(set(first_names + last_names))
    all_names = [name for name in all_names if name not in single_token_cities]

    # Add nouns to the list of cities
    nouns = load_single_token_nouns()
    nouns = [noun for noun in nouns if noun not in single_token_cities and noun not in all_names]
    single_token_cities.extend(nouns)

    # sort to ensure reproducibility
    single_token_cities = sorted(unique(single_token_cities))
    all_names = sorted(unique(all_names))

    # shuffle the names and cities
    rng = random.Random(42)
    rng.shuffle(single_token_cities)
    rng = random.Random(42)
    rng.shuffle(all_names)

    mid = len(all_names) // 2
    e1_names, e2_names = all_names[:mid], all_names[mid : mid * 2]

    print(f"Number of e1 names: {len(e1_names)}")
    print(f"Number of e2 names: {len(e2_names)}")
    print(f"Number of cities: {len(single_token_cities)}")

    # refresh the random seed
    rng = random.Random(42)
    # sample without replacement
    sampled_cities = rng.sample(single_token_cities, k=len(e1_names))
    triplets = [
        (e1, e2, city) for e1, e2, city in zip(e1_names, e2_names, sampled_cities, strict=True)
    ]
    if max_triplets:
        triplets = triplets[:max_triplets]

    # Split triplets into train and test sets
    demoed_triplets, undemoed_triplets = split_triplets(triplets, split_ratio)

    return demoed_triplets, undemoed_triplets


def save_dataset(
    demoed_triplets: List[Tuple],
    undemoed_triplets: List[Tuple],
    max_templates: int | None,
    output_dir: Path,
):
    create_output_files(demoed_triplets, undemoed_triplets, max_templates, output_dir)

    print(f"Dataset generated and saved to {output_dir}")
    # Report detailed numbers
    report_str = f"""
Number of triplets in the dataset: {len(demoed_triplets) + len(undemoed_triplets)}
Number of demoed triplets: {len(demoed_triplets)}
Number of undemoed triplets: {len(undemoed_triplets)}

Maximum number of templates (paraphrases): {max_templates}
    """
    print(report_str)


def generate_dataset(
    output_dir: str | Path,
    split_ratio: float = 0.65,
    max_triplets: int | None = None,
    max_templates: int | None = None,
):
    output_dir = Path(output_dir)
    demoed_triplets, undemoed_triplets = create_dataset(split_ratio, max_triplets)
    save_dataset(demoed_triplets, undemoed_triplets, max_templates, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic spouse dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/synthetic_spouses/processed/all",
        help="Output directory for generated files",
    )
    parser.add_argument("--split_ratio", type=float, default=0.65, help="Train/test split ratio")
    parser.add_argument(
        "--max_triplets", type=int, default=None, help="Number of triplets to max_triplets"
    )
    parser.add_argument(
        "--max_templates", type=int, default=None, help="Maximum number of templates to use"
    )
    args = parser.parse_args()

    generate_dataset(**args.__dict__)


""" 
python latent_reasoning/datagen/synthetic_spouses/generate.py \
    --output_dir datasets/synthetic_spouses/processed/all \
    --split_ratio 0.65
"""
