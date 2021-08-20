/** DROP TABLES **/
DROP TABLE OWNERS CASCADE CONSTRAINTS;
DROP TABLE SPATIALENTITIES CASCADE CONSTRAINTS;
DROP TABLE PROPERTIES CASCADE CONSTRAINTS;
DROP TABLE PICTURES CASCADE CONSTRAINTS;


/** CREATE TABLES **/
CREATE TABLE SpatialEntities(
    id INTEGER NOT NULL PRIMARY KEY,
    name VARCHAR2(32),
    type VARCHAR2(32),
    description VARCHAR2(256),
    geometry SDO_GEOMETRY NOT NULL
);

CREATE TABLE Properties(
    id INTEGER NOT NULL PRIMARY KEY ,
    address VARCHAR2(32),
    property_type VARCHAR2(32),
    price VARCHAR2(32),
    id_owner INTEGER
);

CREATE TABLE Owners (
    id INTEGER NOT NULL PRIMARY KEY,
    name VARCHAR2(32),
    surname VARCHAR2(32),
    email VARCHAR2(32),
    telnum VARCHAR2(32)
);

CREATE TABLE Pictures(
    id INTEGER NOT NULL PRIMARY KEY,
    id_spatial_entity INTEGER NOT NULL,
    is_title INTEGER DEFAULT 0,
    image ORDSYS.ORDImage,
    image_si ORDSYS.SI_StillImage,
    image_ac ORDSYS.SI_AverageColor,
    image_ch ORDSYS.SI_ColorHistogram,
    image_pc ORDSYS.SI_PositionalColor,
    image_tx ORDSYS.SI_Texture
);

/** ALTER TABLES **/
ALTER TABLE Properties ADD CONSTRAINT FK_Properties_Spatial_Entities FOREIGN KEY (id) REFERENCES SpatialEntities;
ALTER TABLE Properties ADD CONSTRAINT FK_Properties_Owners FOREIGN KEY (id_owner) REFERENCES Owners;
ALTER TABLE Pictures ADD CONSTRAINT FK_Pictures_Properties FOREIGN KEY (id_spatial_entity) REFERENCES SpatialEntities;


COMMIT;




/** UPDATE SPATIAL METADATA **/
DELETE FROM USER_SDO_GEOM_METADATA WHERE
        TABLE_NAME = 'SPATIALENTITIES' AND COLUMN_NAME = 'GEOMETRY';

INSERT INTO USER_SDO_GEOM_METADATA VALUES (
    'SpatialEntities', 'geometry', SDO_DIM_ARRAY(SDO_DIM_ELEMENT('X', 0, 800, 0.01), SDO_DIM_ELEMENT('Y', 0, 600, 0.01)), NULL);

DROP INDEX SP_INDEX_entities_geometry;

CREATE INDEX SP_INDEX_entities_geometry ON SpatialEntities ( geometry ) INDEXTYPE IS MDSYS.SPATIAL_INDEX;


COMMIT;




/** INSERT ALL ESTATES **/
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    1, 'A', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 10,10,       100,100   )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    2, 'B', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 110,10,   470,100      )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    3, 'C', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 480,10,      640,100   )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    4, 'D', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 650,10,   690,10,    690,50,     680,50,   680,100,  650,100,   650,10   )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    5, 'E', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 690,10,     790,50 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    6, 'F', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY(  10,110,       100,240)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    7, 'G', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY(  110,110,     400,240  )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    8, 'H', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY(  410,110,    600,240  )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    9, 'I', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY(  610,110,    680,240 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    10, 'J', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 690,250,     790,290)
    )
);


COMMIT;


INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    11, 'K', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 690,170,    790,250)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    12, 'L', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 690,80,      790,170)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    13, 'M', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 690,60,     790,80)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    14, 'N', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 10,340,     80,340,     80,390,     250,390,   250,430,    10,430,     10,340 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    15, 'O', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 10,250,     350,250,    350,430,    260,430,   260,380,    90,380,    90,330,     10,330,   10,250 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    16, 'P', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 360,345,  480,430)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    17, 'Q', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 480,345,   600,430)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    18, 'R', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 360,250,     480,335 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    19, 'S', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 480,250,     600,335 )
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    20, 'T', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 610,250,     680,430 )
    )
);


COMMIT;


INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    21, 'U', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 10,440,     250,590)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    22, 'V', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 260,440,   420,590)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    23, 'W', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 430,440,    680,590)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    24, 'X', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 690,300,   740,300,    740,420,    790,420,   790,590,   690,590,  690,300)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    25, 'Y', 'estate',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 740,300,  790,420)
    )
);

/** INSER ROAD AROUND ESTATES**/
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    26, 'All streets', 'road',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(
            1, 1003, 3,
            5, 2003, 3,
            9, 2003, 3,
            13, 2003, 3,
            17, 2003, 1,
            31, 2003, 3,
            35, 2003, 3,
            39, 2003, 3,
            43, 2003, 3,
            47, 2003, 3,
            51, 2003, 1,
            65, 2003, 1,
            83, 2003, 3,
            87, 2003, 3,
            91, 2003, 3,
            95, 2003, 3,
            99, 2003, 3,
            103, 2003, 3,
            107, 2003, 3

        ),
        SDO_ORDINATE_ARRAY(
            0,0, 800,600,
            10,10, 100,100,
            110,10,     470,100,
            480,10,      640,100,
            650,10,  650,100,  680,100,        680,50,   790,50,  790,10,    650,10,
            10,110,    100,240,
            110,110,   400,240,
            410,110,   600,240,
            610,110,   680,240,
            690,60,    790,290,

            10,340,   10,430,  250,430,         250,390,    80,390,    80,340,     10,340,
            10,250, 10,330, 90,330,     90,380, 260,380,  260,430, 350,430,350,250, 10,250,

            360,345, 600,430,
            360,250, 600,335,
            610,250, 680,430,
            10,440,   250,590,
            260,440, 420,590,
            430,440, 680,590,
            690,300, 790,590
        )
    )
);

COMMIT;


