from app.domains.search.query.models import QueryFeatures, QueryClasses
from app.config.settings import settings

def classify_query(features: QueryFeatures) -> QueryClasses:
    classes = QueryClasses()
    
    # Thresholds (would ideally come from settings)
    TH_LEXICAL_HIGH = getattr(settings, "QUERY_ROUTING_LEXICAL_HIGH", 0.5)
    TH_IDENTIFIER_HIGH = getattr(settings, "QUERY_ROUTING_IDENTIFIER_HIGH", 0.3)
    TH_NL_HIGH = getattr(settings, "QUERY_ROUTING_NL_HIGH", 0.5)
    TH_TECHNICAL_HIGH = getattr(settings, "QUERY_ROUTING_TECHNICAL_HIGH", 0.3)
    
    # 1. Exact Term
    if features.quoted_term_density > 0.5 or (features.token_count <= 2 and features.identifier_density > 0.5):
        classes.EXACT_TERM = True
        
    # 2. Identifier Heavy
    if features.identifier_density > TH_IDENTIFIER_HIGH:
        classes.IDENTIFIER_HEAVY = True
        classes.TECHNICAL = True
        
    # 3. Technical
    # High rarity or identifiers
    if features.term_rarity_score > 0.7 or features.lexical_specificity_score > TH_LEXICAL_HIGH:
        classes.TECHNICAL = True
        
    # 4. Natural Language Question
    if features.natural_language_score > TH_NL_HIGH:
        classes.NATURAL_LANGUAGE_QUESTION = True
        
    # 5. Conceptual
    if features.natural_language_score > 0.3 and features.lexical_specificity_score < 0.3:
        classes.CONCEPTUAL = True
        
    # 6. Ambiguous
    if features.token_count <= 2 and features.lexical_specificity_score < 0.2 and not classes.NATURAL_LANGUAGE_QUESTION:
        classes.AMBIGUOUS = True
        
    # 7. Mixed
    # If it's both highly technical/lexical AND natural language
    if (classes.TECHNICAL or classes.IDENTIFIER_HEAVY) and (classes.NATURAL_LANGUAGE_QUESTION or classes.CONCEPTUAL):
        classes.MIXED = True
        
    # If literally no other class fired, default to mixed
    if not any([classes.EXACT_TERM, classes.IDENTIFIER_HEAVY, classes.TECHNICAL, classes.CONCEPTUAL, classes.NATURAL_LANGUAGE_QUESTION, classes.AMBIGUOUS]):
        classes.MIXED = True
        
    return classes
