KNOWN_CONSENT_XPATHS = [
    "/html/body/div[6]/div/footer/button",
    "/html/body/div[11]/div/div/div/div[1]/div[2]/div/button[2]",
    "/html/body/div[1]/div/div/div/div[2]/div/button[2]",
    "/html/body/div[10]/div[3]/div[3]/div[1]/button[1]",
    "/html/body/div[5]/div/div/div[2]/button[2]",
    "//*[@id='accept-btn']",
    "//*[@id='onesignal-slidedown-cancel-button']",
    "//*[@id='disagree-btn']",
    "//*[@id='com_k2']/div[4]/div[2]/div[2]/div[2]/div[2]/button[1]",
    "/html/body/div[5]/div[2]/div[2]/div[2]/div[2]/button[2]",
    "/html/body/div[5]/div/div/div[2]/button[2]",
    "/html/body/div[1]/div/div[4]/div[1]/div/div[2]/button[4]",
#    '//*[@id="CybotCookiebotDialogBodyButtonDecline"]',
]

# Optional: structured recipes (site/template level) — parsed in the app, passed to fetcher later
# For now we keep only KNOWN_CONSENT_XPATHS as a default list.
def default_consent_xpaths():
    return KNOWN_CONSENT_XPATHS[:]
