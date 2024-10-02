bureau_bal_metadata = {
    "primary_key" : [
        "SK_ID_BUREAU",
    ],
    "detail": [
        "MONTHS_BALANCE",
        "STATUS"
    ],
}

bureau_metada = {
    "primary_key" : [
        "SK_ID_CURR",
    ],
    "prev_key": [
        "SK_ID_BUREAU"
    ],
    "credit_info":[
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "CREDIT_TYPE"  
    ],
    "current_temp_data":[
        "DAYS_CREDIT",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
    ],
    "current_amt_data":[
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY"
    ],
    "history_data":[
        "CNT_CREDIT_PROLONG",
    ]
}

cc_bal_metadata = {
    "primary_key": [
        "SK_ID_CURR", 
    ],
    "secondary_key": [
        "SK_ID_PREV", 
    ],
    "bal_and_credit": [
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE", #AMT_RECIVABLE
        "AMT_TOTAL_RECEIVABLE",
    ],
    "withdraws": [
        "AMT_DRAWINGS_ATM_CURRENT",
        "CNT_DRAWINGS_ATM_CURRENT",
        
        "AMT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        
        "AMT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_OTHER_CURRENT",
        
        "AMT_DRAWINGS_POS_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT"
    ],
    "pmt_inst": [
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM"
    ],
    "status": [
        "MONTHS_BALANCE",
        "NAME_CONTRACT_STATUS", 
        "SK_DPD", 
        "SK_DPD_DEF"
        ]
    }

inst_pmt_metadata = {
    "primary_key": [
        "SK_ID_CURR", 
    ],
    "secondary_key": [
        "SK_ID_PREV", 
    ],
    "detail": [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
}

# consumer loan
pos_cash_bal_metadata = {
    "primary_key": [
        "SK_ID_CURR", 
    ],
    "secondary_key": [
        "SK_ID_PREV", 
    ],
    "detail": [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
}

prev_app_metadata = {
    "primary_key": [
        "SK_ID_CURR", 
    ],
    "secondary_key": [
        "SK_ID_PREV", 
    ],
    "loan_info": [
        "NAME_CONTRACT_TYPE",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_CONTRACT_STATUS",
        "DAYS_DECISION",
        "NAME_PAYMENT_TYPE",
        "CODE_REJECT_REASON",
        "NAME_TYPE_SUITE",
        "NAME_CLIENT_TYPE",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "NFLAG_MICRO_CASH",
        "NFLAG_INSURED_ON_APPROVAL"
    ],
    "application_process": [
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
        "FLAG_LAST_APPL_PER_CONTRACT",
        "NFLAG_LAST_APPL_IN_DAY",
        ]
    ,
    "rates": [
        "RATE_DOWN_PAYMENT",
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED"]
    ,
    "sales_and_payment": [
        "CHANNEL_TYPE",
        "SELLERPLACE_AREA",
        "NAME_SELLER_INDUSTRY",
        "CNT_PAYMENT",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION"]
    ,
    "temporal_info": [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
}

application_metadata = {
    "primary_key" : [
        "SK_ID_CURR",
    ],
    "target":[
        "TARGET",
    ],
    "personal_info": [
        "DAYS_BIRTH",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
    ],
    "family_info":[
        "NAME_FAMILY_STATUS",
        "CNT_FAM_MEMBERS",
        "CNT_CHILDREN",
    ], #4
    "external_info": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ], #3
    
    "loan_info": [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "NAME_CONTRACT_TYPE", #Identification if loan is cash or revolving
        "AMT_GOODS_PRICE", #For consumer loans it is the price of the goods for which the loan is given
    ], #3
    
    "financial_info": [
        "OCCUPATION_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "AMT_INCOME_TOTAL",
        "DAYS_EMPLOYED",
        "ORGANIZATION_TYPE",
        "NAME_HOUSING_TYPE",
        "OWN_CAR_AGE",
    ], #8
    
    "properties_info": [
        "REGION_POPULATION_RELATIVE", #Normalized population of region where client lives (higher number means the client lives in more populated region)
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE",
        "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE",
        "COMMONAREA_MODE",
        "ELEVATORS_MODE",
        "ENTRANCES_MODE",
        "FLOORSMAX_MODE",
        "FLOORSMIN_MODE",
        "LANDAREA_MODE",
        "LIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI",
        "ELEVATORS_MEDI",
        "ENTRANCES_MEDI",
        "FLOORSMAX_MEDI",
        "FLOORSMIN_MEDI",
        "LANDAREA_MEDI",
        "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI",
        "NONLIVINGAPARTMENTS_MEDI",
        "NONLIVINGAREA_MEDI",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "TOTALAREA_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ], #19
    
    "social_info": [
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ], #6
    
    "document_info": [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "HOUR_APPR_PROCESS_START",
        "WEEKDAY_APPR_PROCESS_START",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ], #27

    "reachability": [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "NAME_TYPE_SUITE",
        "FLAG_CONT_MOBILE",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ],
    
    "geo_info": [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY"
    ] #6
}


loan_config = {

    "personal_need_loan": [

        "car_loan",
        "consumer_credit",
        "credit_card",
        "microloan",
        "cash_loan_(non-earmarked)",
        "mobile_operator_loan"

    ],

    "business_loan": [
        "loan_for_working_capital_replenishment",
        "loan_for_business_development",
        "loan_for_the_purchase_of_equipment",
        "loan_for_purchase_of_shares_(margin_lending)"

    ],

    "real_estate_loan": [
        "mortgage",
        "real_estate_loan"

    ],

    "unknown": [
        "another_type_of_loan",
        "unknown_type_of_loan"

    ]

}

loans_config = {
    'car_loan': 'personal_need_loan',
    'consumer_credit': 'personal_need_loan',
    'credit_card': 'personal_need_loan',
    'microloan': 'personal_need_loan',
    'cash_loan_(non-earmarked)': 'personal_need_loan',
    'mobile_operator_loan': 'personal_need_loan',
    'loan_for_working_capital_replenishment': 'business_loan',
    'loan_for_business_development': 'business_loan',
    'loan_for_the_purchase_of_equipment': 'business_loan',
    'loan_for_purchase_of_shares_(margin_lending)': 'business_loan',
    'mortgage': 'real_estate_loan',
    'real_estate_loan': 'real_estate_loan',
    'another_type_of_loan': 'unknown',
    'unknown_type_of_loan': 'unknown'
 }

reg_area_config = {
    "0-0": "reg-home-work",
    "1-0": "reg-home",
    "0-1": "reg-work",
    "1-1": "reg-outside",
}