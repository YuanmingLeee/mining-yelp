CREATE TABLE business
(
    business_id  CHARACTER(22) PRIMARY KEY,
    name         VARCHAR(50)  NOT NULL,
    address      VARCHAR(200) NOT NULL,
    city         VARCHAR(40)  NOT NULL,
    state        CHARACTER(2) NOT NULL,
    postal_code  CHARACTER(6) NOT NULL,
    latitude     FLOAT        NOT NULL,
    longitude    FLOAT        NOT NULL,
    stars        FLOAT        NOT NULL,
    review_count INT          NOT NULL,
    is_open      INT          NOT NULL,
    attributes   TEXT,
    categories   TEXT,
    hours        TEXT
);

CREATE TABLE users
(
    user_id            CHARACTER(22) PRIMARY KEY,
    name               VARCHAR(50) NOT NULL,
    review_count       INT         NOT NULL,
    yelping_since      DATETIME    NOT NULL,
    useful             INT         NOT NULL,
    funny              INT         NOT NULL,
    cool               INT         NOT NULL,
    fans               INT         NOT NULL,
    average_stars      FLOAT       NOT NULL,
    compliment_hot     INT         NOT NULL,
    compliment_more    INT         NOT NULL,
    compliment_profile INT         NOT NULL,
    compliment_cute    INT         NOT NULL,
    compliment_list    INT         NOT NULL,
    compliment_note    INT         NOT NULL,
    compliment_plain   INT         NOT NULL,
    compliment_cool    INT         NOT NULL,
    compliment_funny   INT         NOT NULL,
    compliment_writer  INT         NOT NULL,
    compliment_photos  INT         NOT NULL
);

CREATE TABLE review
(
    review_id   CHARACTER(22) PRIMARY KEY,
    user_id     CHARACTER(22) NOT NULL,
    business_id CHARACTER(22) NOT NULL,
    stars       INT           NOT NULL,
    date        DATETIME      NOT NULL,
    text        TEXT          NOT NULL,
    useful      INT           NOT NULL,
    funny       INT           NOT NULL,
    cool        INT           NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (business_id) REFERENCES business (business_id)
);

CREATE TABLE checkin
(
    business_id CHARACTER(22) PRIMARY KEY,
    date        TEXT,
    FOREIGN KEY (business_id) REFERENCES business (business_id)
);

CREATE TABLE tip
(
    text             TEXT,
    date             DATETIME      NOT NULL,
    compliment_count INT           NOT NULL,
    business_id      CHARACTER(22) NOT NULL,
    user_id          CHARACTER(22) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (business_id) REFERENCES business (business_id)
);

CREATE TABLE user_friends
(
    user_id        CHARACTER(22),
    friend_user_id CHARACTER(22),
    PRIMARY KEY (user_id, friend_user_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (friend_user_id) REFERENCES users (user_id)
);

CREATE TABLE user_entitle
(
    user_id CHARACTER(22),
    entitle INT,
    PRIMARY KEY (user_id, entitle),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
