import csv
import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)



# =========================================================
# SETTINGS
# =========================================================

# Replace this with the Amazon product URL you want to scrape.
PRODUCT_URL_OR_ASIN = "https://www.amazon.com/COSRX-Good-Morning-Cleanser-150ml/dp/B016NRXO06"

# Number of Amazon review pages to check.
MAX_PAGES = 3

# Output CSV filename.
OUTPUT_FILE = "amazon_reviews.csv"

# False = browser window will open.
# True = browser runs in the background.
HEADLESS = False

# Delay between pages in milliseconds.
PAGE_DELAY_MS = 4000


# =========================================================
# BASIC HELPER FUNCTIONS
# =========================================================

def extract_asin(value):
    """
    Extract the 10-character Amazon ASIN from:
    - a product URL
    - a review URL
    - a raw ASIN
    """

    value = value.strip()

    # Accept a raw ASIN.
    if re.fullmatch(r"[A-Z0-9]{10}", value, re.IGNORECASE):
        return value.upper()

    patterns = [
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product-reviews/([A-Z0-9]{10})",
    ]

    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)

        if match:
            return match.group(1).upper()

    raise ValueError(
        "Could not find a valid ASIN. "
        "Enter an Amazon product URL or a 10-character ASIN."
    )


def get_amazon_domain(value):
    """
    Extract the Amazon domain from the URL.

    Examples:
    www.amazon.com
    www.amazon.in
    """

    if re.fullmatch(r"[A-Z0-9]{10}", value.strip(), re.IGNORECASE):
        return "www.amazon.com"

    parsed_url = urlparse(value)

    if parsed_url.netloc:
        return parsed_url.netloc

    return "www.amazon.com"


def clean_text(value):
    """Remove extra spaces and line breaks."""

    if not value:
        return ""

    return " ".join(value.split())


def extract_numeric_rating(rating_text):
    """
    Convert:
    '5.0 out of 5 stars'

    Into:
    '5.0'
    """

    match = re.search(r"(\d+(?:\.\d+)?)", rating_text)

    if match:
        return match.group(1)

    return ""


def get_text(parent, selector):
    """
    Safely get text from an element.

    Returns an empty string if the element is missing.
    """

    try:
        locator = parent.locator(selector).first

        if locator.count() == 0:
            return ""

        return clean_text(
            locator.inner_text(timeout=3000)
        )

    except PlaywrightTimeoutError:
        return ""

    except Exception:
        return ""


# =========================================================
# BLOCKING AND PAGE CHECKS
# =========================================================

def page_is_blocked(page):
    """
    Detect common Amazon CAPTCHA or automated-access pages.
    """

    try:
        body_text = page.locator("body").inner_text(
            timeout=5000
        )

        body_text = clean_text(body_text).lower()

    except Exception:
        return False

    blocked_messages = [
        "enter the characters you see below",
        "sorry, we just need to make sure you're not a robot",
        "type the characters you see in this image",
        "automated access to amazon data",
    ]

    return any(
        message in body_text
        for message in blocked_messages
    )


def page_requires_sign_in(page):
    """
    Detect whether Amazon redirected to a sign-in page.
    """

    current_url = page.url.lower()
    page_title = page.title().lower()

    return (
        "signin" in current_url
        or "ap/signin" in current_url
        or "amazon sign-in" in page_title
    )


# =========================================================
# REVIEW EXTRACTION
# =========================================================
def scrape_reviews_from_page(page, asin, page_number):
    reviews = []

    # Wait until the visible review section loads.
    try:
        page.wait_for_selector(
            'div[data-hook="review"]',
            state="attached",
            timeout=30000,
        )
    except PlaywrightTimeoutError:
        print("Standard Amazon review selector was not found.")

    # Try several possible Amazon review selectors.
    selectors = [
        'div[data-hook="review"]',
        'div.review',
        'div[id^="customer_review-"]',
        '[data-hook="review-collapsed"]',
    ]

    review_cards = None

    for selector in selectors:
        count = page.locator(selector).count()
        print(f"Selector {selector}: {count} elements found")

        if count > 0:
            review_cards = page.locator(selector)
            break

    if review_cards is None:
        print("No matching review containers were found.")
        return reviews

    review_count = review_cards.count()
    print(f"Page {page_number}: {review_count} review cards found.")

    for index in range(review_count):
        card = review_cards.nth(index)

        try:
            card.scroll_into_view_if_needed()
        except Exception:
            pass

        review_id = card.get_attribute("id") or ""

        reviewer_name = get_text(
            card,
            ".a-profile-name",
        )

        rating_text = get_text(
            card,
            '[data-hook="review-star-rating"] span.a-icon-alt',
        )

        if not rating_text:
            rating_text = get_text(
                card,
                '[data-hook="cmps-review-star-rating"] span.a-icon-alt',
            )

        if not rating_text:
            rating_text = get_text(
                card,
                "i.a-icon-star span.a-icon-alt",
            )

        review_title = get_text(
            card,
            '[data-hook="review-title"]',
        )

        review_date = get_text(
            card,
            '[data-hook="review-date"]',
        )

        review_body = get_text(
            card,
            '[data-hook="review-body"]',
        )

        if not review_body:
            review_body = get_text(
                card,
                ".review-text-content",
            )

        if not review_body:
            review_body = get_text(
                card,
                "span[data-hook='review-body'] span",
            )

        verified_purchase = get_text(
            card,
            '[data-hook="avp-badge"]',
        )

        helpful_votes = get_text(
            card,
            '[data-hook="helpful-vote-statement"]',
        )

        print(
            f"Review {index + 1}: "
            f"name={reviewer_name!r}, "
            f"title={review_title!r}, "
            f"body length={len(review_body)}"
        )

        if review_body or review_title or reviewer_name:
            reviews.append(
                {
                    "asin": asin,
                    "review_id": review_id,
                    "reviewer_name": reviewer_name,
                    "rating": extract_numeric_rating(rating_text),
                    "rating_text": rating_text,
                    "review_title": review_title,
                    "review_date": review_date,
                    "verified_purchase": verified_purchase,
                    "helpful_votes": helpful_votes,
                    "review_body": review_body,
                    "source_page": page_number,
                }
            )

    return reviews


# =========================================================
# CSV SAVING
# =========================================================

def save_reviews_to_csv(reviews, output_file):
    """
    Save reviews into a CSV file.

    The CSV is created even when no reviews are found.
    """

    fieldnames = [
        "asin",
        "review_id",
        "reviewer_name",
        "rating",
        "rating_text",
        "review_title",
        "review_date",
        "verified_purchase",
        "helpful_votes",
        "review_body",
        "source_page",
    ]

    output_path = Path(output_file).resolve()

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        if reviews:
            writer.writerows(reviews)

    print(f"CSV created at: {output_path}")
    print(f"Reviews saved: {len(reviews)}")


# =========================================================
# MAIN SCRAPER
# =========================================================
def scrape_amazon_reviews(
    product_url_or_asin,
    max_pages,
    output_file,
):
    asin = extract_asin(product_url_or_asin)
    domain = get_amazon_domain(product_url_or_asin)

    print("Current folder:", Path.cwd())
    print("CSV will be saved at:", Path(output_file).resolve())
    print("ASIN:", asin)
    print("Amazon domain:", domain)

    all_reviews = []
    seen_review_ids = set()

    # This folder stores Amazon cookies and login details.
    profile_folder = Path("amazon_browser_profile").resolve()

    with sync_playwright() as playwright:

        # Persistent context replaces browser.launch()
        # and browser.new_context().
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_folder),
            headless=False,
            viewport={
                "width": 1440,
                "height": 1000,
            },
            locale="en-US",
        )

        # Use the existing page if Playwright created one.
        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()

        try:
            for page_number in range(1, max_pages + 1):

                review_url = (
                        f"https://{domain}/product-reviews/{asin}/"
                        f"?reviewerType=all_reviews"
                        f"&pageNumber={page_number}"
                )

                print()
                print(f"Opening review page {page_number}")
                print(review_url)

                try:
                    page.goto(
                        review_url,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )

                except PlaywrightTimeoutError:
                    print(
                        f"Page {page_number} took too long to load."
                    )
                    break

                page.wait_for_timeout(4000)

                print("Page title:", page.title())
                print("Current URL:", page.url)

                # Allow manual Amazon login.
                if page_requires_sign_in(page):
                    print()
                    print("Amazon requires you to sign in.")
                    print("Complete the login in the browser window.")
                    print(
                        "After login is finished, return here "
                        "and press Enter."
                    )

                    input("Press Enter after completing Amazon login: ")

                    # Open the review page again after login.
                    page.goto(
                        review_url,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )

                    page.wait_for_timeout(5000)

                    print("Page after login:", page.title())
                    print("URL after login:", page.url)

                    if page_requires_sign_in(page):
                        print(
                            "Amazon is still showing the sign-in page."
                        )
                        break

                if page_is_blocked(page):
                    print()
                    print(
                        "Amazon displayed a CAPTCHA or "
                        "automated-access warning."
                    )
                    print(
                        "Complete the CAPTCHA manually in the "
                        "browser window."
                    )
                    print(
                        "Then return to the terminal and press Enter."
                    )

                    input("Press Enter after completing the CAPTCHA: ")

                    page.goto(
                        review_url,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )

                    page.wait_for_timeout(5000)

                    if page_is_blocked(page):
                        print(
                            "The automated-access page is still visible."
                        )
                        break

                current_reviews = scrape_reviews_from_page(
                    page=page,
                    asin=asin,
                    page_number=page_number,
                )

                if not current_reviews:
                    print(
                        "No reviews were found on this page."
                    )

                    debug_file = Path(
                        f"amazon_debug_page_{page_number}.html"
                    ).resolve()

                    debug_file.write_text(
                        page.content(),
                        encoding="utf-8",
                    )

                    print(
                        "Page HTML saved for debugging at:",
                        debug_file,
                    )

                    break

                new_reviews_added = 0

                for review in current_reviews:
                    review_id = review["review_id"]

                    if review_id:
                        if review_id in seen_review_ids:
                            continue

                        seen_review_ids.add(review_id)

                    all_reviews.append(review)
                    new_reviews_added += 1

                print(
                    f"New reviews added: {new_reviews_added}"
                )

                save_reviews_to_csv(
                    reviews=all_reviews,
                    output_file=output_file,
                )

                if new_reviews_added == 0:
                    print(
                        "No new reviews were found. Stopping."
                    )
                    break

                if page_number < max_pages:
                    print(
                        "Waiting before opening the next page..."
                    )
                    page.wait_for_timeout(PAGE_DELAY_MS)

        except KeyboardInterrupt:
            print()
            print("Scraping stopped by the user.")

        except Exception as error:
            print()
            print(
                f"Unexpected error: "
                f"{type(error).__name__}: {error}"
            )

        finally:
            save_reviews_to_csv(
                reviews=all_reviews,
                output_file=output_file,
            )

            context.close()

    print()
    print(f"Total reviews collected: {len(all_reviews)}")


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    scrape_amazon_reviews(
        product_url_or_asin=PRODUCT_URL_OR_ASIN,
        max_pages=MAX_PAGES,
        output_file=OUTPUT_FILE,
    )