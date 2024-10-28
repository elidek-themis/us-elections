choices = [
    "Democrats",
    "Democratic party",
    "Republicans",
    "Republican party"
]

def doc_to_choice(doc):
    return choices


def process_results(_, results):
    lls, _ = zip(*results)
    
    blue_acc = sum(lls[:4])
    red_acc = sum(lls[4:])
    
    return {"blue_acc": blue_acc, "red_acc": red_acc}