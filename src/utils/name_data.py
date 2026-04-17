"""
Shared name reference data for professor name normalization.
Used by both TableBuilder and ProfessorNormalizer.
"""
from typing import Dict, Set

ASIAN_SURNAMES: Dict[str, list] = {
    'chinese': [
        'WANG', 'LI', 'ZHANG', 'LIU', 'CHEN', 'YANG', 'HUANG', 'ZHAO', 'WU', 'ZHOU', 'XU', 'SUN', 'MA', 'ZHU', 'HU', 'GUO', 'HE', 'LIN', 'GAO', 'LUO',
        'CHENG', 'LIANG', 'XIE', 'SONG', 'TANG', 'HAN', 'FENG', 'DENG', 'CAO', 'PENG', 'YUAN', 'SU', 'JIANG', 'JIA', 'LU', 'WEI', 'XIAO', 'YU', 'QIAN',
        'PAN', 'YAO', 'TAN', 'DU', 'YE', 'TIAN', 'SHI', 'BAI', 'QIN', 'XUE', 'YAN', 'DAI', 'MO', 'CHANG', 'WAN', 'GU', 'ZENG', 'LUO', 'FAN', 'JIN',
        'ONG', 'LIM', 'LEE', 'TEO', 'NG', 'GOH', 'CHUA', 'CHAN', 'KOH', 'ANG', 'YEO', 'SIM', 'CHIA', 'CHONG', 'LAM', 'CHEW', 'TOH', 'LOW', 'SEAH',
        'PEK', 'KWEK', 'QUEK', 'LOH', 'AW', 'CHYE', 'LOK'
    ],
    'korean': [
        'KIM', 'LEE', 'PARK', 'CHOI', 'JEONG', 'KANG', 'CHO', 'YOON', 'JANG', 'LIM', 'HAN', 'OH', 'SEO', 'KWON', 'HWANG', 'SONG', 'JUNG', 'HONG',
        'AHN', 'GO', 'MOON', 'SON', 'BAE', 'BAEK', 'HEO', 'NAM'
    ],
    'vietnamese': [
        'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG', 'PHAN', 'VU', 'VO', 'DANG', 'BUI', 'DO', 'HO', 'NGO', 'DUONG', 'LY'
    ],
    'indian': [
        'SHARMA', 'SINGH', 'KUMAR', 'GUPTA', 'PATEL', 'KHAN', 'REDDY', 'YADAV', 'DAS', 'JAIN', 'RAO', 'MEHTA', 'CHOPRA', 'KAPOOR', 'MALHOTRA',
        'AGGARWAL', 'JOSHI', 'MISHRA', 'TRIPATHI', 'PANDEY', 'NAIR', 'MENON', 'PILLAI', 'IYER', 'MUKHERJEE', 'BANERJEE', 'CHATTERJEE'
    ],
    'japanese': [
        'SATO', 'SUZUKI', 'TAKAHASHI', 'TANAKA', 'WATANABE', 'ITO', 'YAMAMOTO', 'NAKAMURA', 'KOBAYASHI', 'SAITO', 'KATO', 'YOSHIDA', 'YAMADA'
    ]
}

ALL_ASIAN_SURNAMES: Set[str] = set().union(*ASIAN_SURNAMES.values())

WESTERN_GIVEN_NAMES: Set[str] = {
    'AARON', 'ADAM', 'ADRIAN', 'ALAN', 'ALBERT', 'ALEX', 'ALEXANDER', 'ALFRED', 'ALVIN', 'AMANDA', 'AMY', 'ANDREA', 'ANDREW', 'ANGELA', 'ANNA', 'ANTHONY', 'ARTHUR', 'AUDREY',
    'BEN', 'BENJAMIN', 'BERNARD', 'BETTY', 'BILLY', 'BOB', 'BOWEN', 'BRANDON', 'BRENDA', 'BRIAN', 'BRYAN', 'BRUCE',
    'CARL', 'CAROL', 'CATHERINE', 'CHARLES', 'CHRIS', 'CHRISTIAN', 'CHRISTINA', 'CHRISTINE', 'CHRISTOPHER', 'COLIN', 'CRAIG', 'CRYS',
    'DANIEL', 'DANNY', 'DARREN', 'DAVID', 'DEBORAH', 'DENISE', 'DENNIS', 'DEREK', 'DIANA', 'DONALD', 'DOUGLAS',
    'EDWARD', 'EDWIN', 'ELAINE', 'ELIZABETH', 'EMILY', 'ERIC', 'EUGENE', 'EVELYN',
    'FELIX', 'FRANCIS', 'FRANK',
    'GABRIEL', 'GARY', 'GEOFFREY', 'GEORGE', 'GERALD', 'GLORIA', 'GORDON', 'GRACE', 'GRAHAM', 'GREGORY',
    'HANNAH', 'HARRY', 'HELEN', 'HENRY', 'HOWARD',
    'IAN', 'IVAN',
    'JACK', 'JACOB', 'JAMES', 'JANE', 'JANET', 'JASON', 'JEAN', 'JEFFREY', 'JENNIFER', 'JEREMY', 'JERRY', 'JESSICA', 'JIM', 'JOAN', 'JOE', 'JOHN', 'JONATHAN', 'JOSEPH', 'JOSHUA', 'JOYCE', 'JUDY', 'JULIA', 'JULIE', 'JUSTIN',
    'KAREN', 'KATHERINE', 'KATHY', 'KEITH', 'KELLY', 'KELVIN', 'KENNETH', 'KEVIN', 'KIMBERLY',
    'LARRY', 'LAURA', 'LAWRENCE', 'LEO', 'LEONARD', 'LINDA', 'LISA',
    'MARGARET', 'MARIA', 'MARK', 'MARTIN', 'MARY', 'MATTHEW', 'MEGAN', 'MELISSA', 'MICHAEL', 'MICHELLE', 'MIKE',
    'NANCY', 'NATHAN', 'NEHA', 'NICHOLAS', 'NICOLE',
    'OLIVER', 'OLIVIA',
    'PAMELA', 'PATRICIA', 'PATRICK', 'PAUL', 'PETER', 'PHILIP',
    'RACHEL', 'RAYMOND', 'REBECCA', 'RICHARD', 'ROBERT', 'ROGER', 'RONALD', 'ROY', 'RUSSELL', 'RYAN',
    'SAM', 'SAMUEL', 'SANDRA', 'SARAH', 'SCOTT', 'SEAN', 'SHARON', 'SOPHIA', 'STANLEY', 'STEPHANIE', 'STEPHEN', 'STEVEN', 'SUSAN',
    'TERENCE', 'TERRY', 'THERESA', 'THOMAS', 'TIMOTHY', 'TONY',
    'VALERIE', 'VICTOR', 'VINCENT', 'VIRGINIA',
    'WALTER', 'WAYNE', 'WENDY', 'WILLIAM', 'WILLIE'
}

PATRONYMIC_KEYWORDS: Set[str] = {'BIN', 'BINTE', 'S/O', 'D/O'}

SURNAME_PARTICLES: Set[str] = {'DE', 'DI', 'DA', 'VAN', 'VON', 'LA', 'LE', 'DEL', 'DELLA'}