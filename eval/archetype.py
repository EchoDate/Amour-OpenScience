big5_mapping = {
  "A1_Analyst": {
    "label": "Rational Analyst",
    "openness": "medium",
    "conscientiousness": "high",
    "extraversion": "low",
    "agreeableness": "medium",
    "neuroticism": "low"
  },
  "A2_RiskAversePlanner": {
    "label": "Risk-Averse Planner",
    "openness": "low",
    "conscientiousness": "high",
    "extraversion": "low",
    "agreeableness": "low",
    "neuroticism": "high"
  },
  "A3_CalmManager": {
    "label": "Calm Manager",
    "openness": "medium",
    "conscientiousness": "high",
    "extraversion": "medium",
    "agreeableness": "medium",
    "neuroticism": "low"
  },
  "A4_Technocrat": {
    "label": "Technocratic Type",
    "openness": "low",
    "conscientiousness": "high",
    "extraversion": "low",
    "agreeableness": "low",
    "neuroticism": "low"
  },
  "B1_Persuader": {
    "label": "Extroverted Persuader",
    "openness": "high",
    "conscientiousness": "medium",
    "extraversion": "high",
    "agreeableness": "medium",
    "neuroticism": "low"
  },
  "B2_Advocate": {
    "label": "Enthusiastic Advocate",
    "openness": "high",
    "conscientiousness": "medium",
    "extraversion": "high",
    "agreeableness": "high",
    "neuroticism": "medium"
  },
  "B3_DominantLeader": {
    "label": "Leader Type",
    "openness": "medium",
    "conscientiousness": "high",
    "extraversion": "high",
    "agreeableness": "low",
    "neuroticism": "low"
  },
  "B4_CharismaticRiskTaker": {
    "label": "Charismatic Risk-Taker",
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "high",
    "agreeableness": "medium",
    "neuroticism": "low"
  },
  "C1_Empath": {
    "label": "Empathetic Listener",
    "openness": "high",
    "conscientiousness": "medium",
    "extraversion": "low",
    "agreeableness": "high",
    "neuroticism": "medium"
  },
  "C2_Mediator": {
    "label": "Emotional Mediator",
    "openness": "medium",
    "conscientiousness": "high",
    "extraversion": "medium",
    "agreeableness": "high",
    "neuroticism": "low"
  },
  "C3_AnxiousSupportSeeker": {
    "label": "Attachment Type",
    "openness": "medium",
    "conscientiousness": "low",
    "extraversion": "medium",
    "agreeableness": "high",
    "neuroticism": "high"
  },
  "C4_OverAccommodating": {
    "label": "Over-Empathetic Type",
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "low",
    "agreeableness": "high",
    "neuroticism": "high"
  },
  "D1_DefensiveSkeptic": {
    "label": "Defensive Skeptic",
    "openness": "low",
    "conscientiousness": "medium",
    "extraversion": "low",
    "agreeableness": "low",
    "neuroticism": "high"
  },
  "D2_PassiveAggressive": {
    "label": "Passive-Aggressive Type",
    "openness": "medium",
    "conscientiousness": "medium",
    "extraversion": "low",
    "agreeableness": "low",
    "neuroticism": "high"
  },
  "D3_Confrontational": {
    "label": "Confrontational Type",
    "openness": "low",
    "conscientiousness": "high",
    "extraversion": "high",
    "agreeableness": "low",
    "neuroticism": "medium"
  },
  "D4_Detached": {
    "label": "Detached Type",
    "openness": "low",
    "conscientiousness": "low",
    "extraversion": "low",
    "agreeableness": "low",
    "neuroticism": "low"
  },
  "E1_Volatile": {
    "label": "Emotionally Volatile",
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "high",
    "agreeableness": "low",
    "neuroticism": "high"
  },
  "E2_IdealisticUnstable": {
    "label": "Idealistic but Unstable",
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "medium",
    "agreeableness": "high",
    "neuroticism": "high"
  },
  "E3_Burnout": {
    "label": "Stress-Breakdown Type",
    "openness": "medium",
    "conscientiousness": "high",
    "extraversion": "low",
    "agreeableness": "medium",
    "neuroticism": "high"
  },
  "E4_Impulsive": {
    "label": "Impulsive Type",
    "openness": "high",
    "conscientiousness": "low",
    "extraversion": "high",
    "agreeableness": "low",
    "neuroticism": "medium"
  }
}


initial_state_mapping = {
  "A1_Analyst": { "stress": 0.2, "trust": 0.47, "dominance": 0.17, "focus": 0.62 },
  "A2_RiskAversePlanner": { "stress": 0.56, "trust": 0.15, "dominance": 0.2, "focus": 0.45 },
  "A3_CalmManager": { "stress": 0.2, "trust": 0.47, "dominance": 0.32, "focus": 0.62 },
  "A4_Technocrat": { "stress": 0.21, "trust": 0.3, "dominance": 0.25, "focus": 0.6 },
  "B1_Persuader": { "stress": 0.2, "trust": 0.5, "dominance": 0.47, "focus": 0.55 },
  "B2_Advocate": { "stress": 0.36, "trust": 0.62, "dominance": 0.4, "focus": 0.5 },
  "B3_DominantLeader": { "stress": 0.21, "trust": 0.3, "dominance": 0.55, "focus": 0.6 },
  "B4_CharismaticRiskTaker": { "stress": 0.26, "trust": 0.5, "dominance": 0.47, "focus": 0.3 },
  "C1_Empath": { "stress": 0.36, "trust": 0.62, "dominance": 0.17, "focus": 0.5 },
  "C2_Mediator": { "stress": 0.19, "trust": 0.62, "dominance": 0.25, "focus": 0.6 },
  "C3_AnxiousSupportSeeker": { "stress": 0.64, "trust": 0.43, "dominance": 0.15, "focus": 0.2 },
  "C4_OverAccommodating": { "stress": 0.63, "trust": 0.45, "dominance": 0.02, "focus": 0.25 },
  "D1_DefensiveSkeptic": { "stress": 0.56, "trust": 0.15, "dominance": 0.2, "focus": 0.4 },
  "D2_PassiveAggressive": { "stress": 0.55, "trust": 0.18, "dominance": 0.25, "focus": 0.42 },
  "D3_Confrontational": { "stress": 0.38, "trust": 0.08, "dominance": 0.5, "focus": 0.38 },
  "D4_Detached": { "stress": 0.31, "trust": 0.23, "dominance": 0.2, "focus": 0.3 },
  "E1_Volatile": { "stress": 0.66, "trust": 0.2, "dominance": 0.5, "focus": 0.2 },
  "E2_IdealisticUnstable": { "stress": 0.64, "trust": 0.5, "dominance": 0.2, "focus": 0.2 },
  "E3_Burnout": { "stress": 0.55, "trust": 0.33, "dominance": 0.12, "focus": 0.47 },
  "E4_Impulsive": { "stress": 0.49, "trust": 0.28, "dominance": 0.52, "focus": 0.28 }
}

