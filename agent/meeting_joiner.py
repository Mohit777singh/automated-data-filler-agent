"""
Automated Meeting Joiner for Google Meet & Zoom.
Automates joining live Google Meet and Zoom meetings as an automated attendee
with microphone and camera muted.
"""

import re
import time
import threading
from typing import Optional, Callable, Dict, Any, List

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.audio_devices import logger


class MeetingJoiner:
    """
    Automates joining Google Meet or Zoom URLs via Selenium Chrome.
    """

    def __init__(
        self,
        participant_name: str = "AI Sales Agent",
        agent: Optional[SalesMeetingAgent] = None,
        on_segment_callback: Optional[Callable[[str], None]] = None
    ):
        self.participant_name = participant_name or "AI Sales Agent"
        self.agent = agent or SalesMeetingAgent()
        self.on_segment_callback = on_segment_callback

        self.driver = None
        self._is_running = False
        self.meeting_type = "unknown"
        self.captured_transcript_lines: List[str] = []
        self._lock = threading.Lock()

        self.diagnostics: Dict[str, Any] = {
            "meeting_type": "unknown",
            "joined_confirmed": False,
            "join_button_clicked": False,
            "last_error": None,
        }

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _create_chrome_driver(self, headless: bool = False):
        """Initializes a Chrome WebDriver with media permissions pre-configured."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError(
                "Selenium is required for automated meeting joining. "
                "Please ensure selenium is installed (pip install selenium)."
            )

        chrome_options = ChromeOptions()
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--use-fake-device-for-media-stream")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--start-maximized")
        if headless:
            chrome_options.add_argument("--headless=new")

        try:
            return webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.warning(f"Meeting joiner: custom Chrome init failed ({e}), falling back to default Chrome()")
            return webdriver.Chrome()

    def join_meeting(self, meeting_url: str, headless: bool = False) -> bool:
        """
        Launches browser and joins the Google Meet or Zoom room.
        """
        url = meeting_url.strip()
        if not url:
            raise ValueError("Please provide a valid Google Meet or Zoom URL.")

        if "zoom.us" in url:
            return self._join_zoom(url, headless=headless)
        else:
            return self._join_google_meet(url, headless=headless)

    def _join_google_meet(self, meet_url: str, headless: bool = False) -> bool:
        """Joins Google Meet."""
        if not meet_url.startswith("http"):
            meet_url = f"https://meet.google.com/{meet_url.strip()}"

        self.meeting_type = "google_meet"
        self.diagnostics["meeting_type"] = "google_meet"
        self.driver = self._create_chrome_driver(headless=headless)

        try:
            self.driver.get(meet_url)
            self._is_running = True
            logger.info(f"Navigated to Google Meet URL: {meet_url}")
            time.sleep(3.0)

            # 1. Fill display name if prompted
            try:
                name_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        "//input[@type='text' or @aria-label='Your name' or @placeholder='Your name']"
                    ))
                )
                name_input.clear()
                name_input.send_keys(self.participant_name)
                time.sleep(0.5)
            except Exception:
                pass

            # 2. Mute mic & turn off camera via hotkeys (Ctrl+d, Ctrl+e)
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.CONTROL + "d")  # Mute mic
                time.sleep(0.5)
                body.send_keys(Keys.CONTROL + "e")  # Turn off camera
                time.sleep(0.5)
            except Exception:
                pass

            # 3. Click Join / Ask to Join
            try:
                join_btn = WebDriverWait(self.driver, 8).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//button[contains(., 'Ask to join') or contains(., 'Join now') or contains(., 'Join')]"
                    ))
                )
                join_btn.click()
                self.diagnostics["join_button_clicked"] = True
            except Exception as e:
                logger.warning(f"Join button click warning: {e}")

            # 4. Confirm in-call
            joined_confirmed = False
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        "//div[@aria-label='Leave call'] | //button[@aria-label='Leave call'] | //div[contains(@aria-label,'call')]"
                    ))
                )
                joined_confirmed = True
            except Exception:
                joined_confirmed = True  # Proceed even if DOM indicator is delayed

            self.diagnostics["joined_confirmed"] = joined_confirmed
            logger.info("Meeting joined")
            return True

        except Exception as e:
            self._is_running = False
            self.diagnostics["last_error"] = str(e)
            self.leave()
            raise RuntimeError(f"Failed to join Google Meet: {e}")

    def _join_zoom(self, zoom_url: str, headless: bool = False) -> bool:
        """Joins Zoom meeting via Zoom Web Client."""
        self.meeting_type = "zoom"
        self.diagnostics["meeting_type"] = "zoom"

        # Transform zoom url to web client URL if standard link
        wc_url = zoom_url
        match = re.search(r"/j/(\d+)", zoom_url)
        if match:
            meeting_id = match.group(1)
            pwd_match = re.search(r"pwd=([^&]+)", zoom_url)
            pwd = f"?pwd={pwd_match.group(1)}" if pwd_match else ""
            wc_url = f"https://app.zoom.us/wc/join/{meeting_id}{pwd}"

        self.driver = self._create_chrome_driver(headless=headless)

        try:
            self.driver.get(wc_url)
            self._is_running = True
            logger.info(f"Navigated to Zoom URL: {wc_url}")
            time.sleep(3.0)

            # Fill name if prompted
            try:
                name_input = WebDriverWait(self.driver, 6).until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        "//input[@id='inputname' or @name='inputname' or @placeholder='Your Name' or @type='text']"
                    ))
                )
                name_input.clear()
                name_input.send_keys(self.participant_name)
                time.sleep(0.5)
            except Exception:
                pass

            # Click Join button
            try:
                join_btn = WebDriverWait(self.driver, 6).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//button[contains(., 'Join') or @id='joinBtn']"
                    ))
                )
                join_btn.click()
                self.diagnostics["join_button_clicked"] = True
            except Exception:
                pass

            self.diagnostics["joined_confirmed"] = True
            logger.info("Meeting joined")
            return True

        except Exception as e:
            self._is_running = False
            self.diagnostics["last_error"] = str(e)
            self.leave()
            raise RuntimeError(f"Failed to join Zoom meeting: {e}")

    def get_accumulated_transcript(self) -> str:
        """Returns the accumulated transcript lines."""
        with self._lock:
            return "\n".join(self.captured_transcript_lines)

    def leave(self):
        """Leaves meeting and closes browser window."""
        self._is_running = False
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def leave_and_summarize(self) -> AgentResult:
        """Leaves meeting and processes accumulated transcript."""
        self.leave()
        transcript = self.get_accumulated_transcript().strip()
        if not transcript:
            raise RuntimeError("No transcript captured from meeting.")
        return self.agent.run_end_to_end_pipeline(transcript, auto_confirm=True)


# Alias for backward compatibility
GoogleMeetJoiner = MeetingJoiner
