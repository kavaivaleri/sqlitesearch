"""
FAQ Search using sqlitesearch

This script demonstrates using sqlitesearch to build a simple FAQ search
for DataTalks Club courses.
"""

import requests
from sqlitesearch import TextSearchIndex


def main():
    # Fetch the courses index
    base_faq_url = 'https://datatalks.club/faq'
    courses_index_url = f'{base_faq_url}/json/courses.json'

    print("Fetching courses index...")
    courses_index = requests.get(courses_index_url).json()
    print(f"Found {len(courses_index)} courses:")
    for course in courses_index:
        print(f"  - {course['course_name']}: {course['questions_count']} questions")

    # Load all FAQ documents
    print("\nLoading FAQ documents...")
    documents = []

    for course in courses_index:
        course_path = course['path']
        course_url = f'{base_faq_url}/{course_path}'
        course_data = requests.get(course_url).json()
        documents.extend(course_data)

    print(f"Loaded {len(documents)} total documents")

    # Create the search index
    db_path = 'faq.db'
    print(f"\nCreating search index at {db_path}...")

    index = TextSearchIndex(
        text_fields=['section', 'question', 'answer'],
        keyword_fields=['course'],
        db_path=db_path
    )

    # Check if we need to load data (index might already exist)
    import os
    if os.path.exists(db_path) and not index._is_empty():
        print("Index already exists with data. Reusing existing index.")
        print("Delete faq.db to rebuild the index.")
    else:
        print("Indexing documents...")
        index.fit(documents)
        print("Indexing complete!")

    # Example searches
    print("\n" + "=" * 60)
    print("EXAMPLE SEARCHES")
    print("=" * 60)

    questions = [
        "I just discovered the course. Can I join now?",
        "How do I get a certificate?",
        "What are the prerequisites?",
        "homework deadline",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        filter_dict = {'course': 'llm-zoomcamp'}
        boost_dict = {'question': 3, 'section': 0.5}

        results = index.search(
            question,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=3
        )

        if results:
            for i, r in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Section: {r.get('section', 'N/A')}")
                print(f"    Question: {r.get('question', 'N/A')[:80]}...")
                print(f"    Answer: {r.get('answer', 'N/A')[:120]}...")
        else:
            print("  No results found.")

    # Demonstrate persistence
    print("\n" + "=" * 60)
    print("PERSISTENCE DEMONSTRATION")
    print("=" * 60)
    print(f"\nThe index is saved at: {db_path}")
    print("You can reopen it later with:")
    print(f"  index = TextSearchIndex(")
    print(f"      text_fields=['section', 'question', 'answer'],")
    print(f"      keyword_fields=['course'],")
    print(f"      db_path='{db_path}'")
    print(f"  )")

    index.close()


if __name__ == "__main__":
    main()
