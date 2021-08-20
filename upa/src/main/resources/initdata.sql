-- shops
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    27, 'Tesco', 'shop',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 10,500, 100,590)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    28, 'Billa', 'shop',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 10,250, 90,330)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    29, 'Lidl', 'shop',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 210,10, 350,50)
    )
);

-- park
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    30, 'Park kultury', 'park',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 200,150, 300,200)
    )
);

-- schools
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    31, 'Gymnazium L.S.', 'school',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 290,340, 330,340, 330,350, 350,350, 350,400, 330,400, 330,410, 290,410, 290,340)
    )
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    32, 'Zakladna skola', 'school',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 720,170, 790,250 )
    )
);

--hospital
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    33, 'Nemocnica sv.Jakuba', 'hospital',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 1),
        SDO_ORDINATE_ARRAY( 430,520, 470,520, 470,460, 520,460, 520,590, 430,590, 430,520 )
    )
);


-- trees
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    34, 'Stromy pri Tescu', 'trees',
    SDO_GEOMETRY (2005, null, null,
	SDO_ELEM_INFO_ARRAY (1,1,1, 3,1,1, 5,1,1),
    	SDO_ORDINATE_ARRAY (120,550, 140,550, 130,530))
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    35, 'Stromy pri nemocnici', 'trees',
    SDO_GEOMETRY (2005, null, null,
	SDO_ELEM_INFO_ARRAY (1,1,1, 3,1,1, 5,1,1),
    	SDO_ORDINATE_ARRAY (450,460 ,455,470, 445,455))
);

COMMIT;

-- tram line 12, spodna linka
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    36, 'Technologicky park 12', 'tram line',
    SDO_GEOMETRY(2002, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1,4,3, 1,2,1, 3,2,1, 5,2,1),
        SDO_ORDINATE_ARRAY( 10,105,  685,105,  685,55,  790,55)
    )
);

-- tram line 6, horna linka
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    40, 'Komarov 6', 'tram line',
    SDO_GEOMETRY(2002, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1,4,2, 1,2,1, 3,2,1),
        SDO_ORDINATE_ARRAY( 10,435,  685,435,  685,590)
    )
);

-- tram line 1, zvisla linka
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES
    (43, 'Reckovice 1', 'tram line',
    SDO_GEOMETRY(2002, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1,4,7, 1,2,1, 3,2,1, 5,2,1,  7,2,1,  9,2,1,   11,2,1,   13,2,1),
        SDO_ORDINATE_ARRAY( 425,590, 425,435,  355,435,  355,245,  405,245,  405, 105,   475,105,  475,10 )
    )
);

COMMIT;

-- tram stops
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    37, 'TL12 Klusackova', 'tram stop',
    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(120, 105, NULL), NULL, NULL)
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    38, 'TL12 Nerudova', 'tram stop',
        SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(450, 105, NULL), NULL,NULL)
);
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    39, 'TL12 Ceska', 'tram stop',
    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(660, 105, NULL), NULL,NULL)
);
-- tram stops:
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    41, 'TL6 Mojmirova', 'tram stop',
    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(410, 435, NULL), NULL,NULL)
);

INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    42, 'TL6 Kulata', 'tram stop',
    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(670, 435, NULL), NULL,NULL)
);

-- tram stops:
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    44, 'TL1 Semilasso', 'tram stop',
    SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(390, 245, NULL), NULL, NULL)
);

COMMIT;

-- owners
INSERT INTO Owners VALUES (
    1, 'Lukas', 'Smetana', 'lukas.smetana@gmail.com','475562311'
);


INSERT INTO Owners VALUES (
    2, 'Milan', 'Prikrop', 'm.prikrop@gmail.com','472133645'
);

-- houses
INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
    45, 'Dom na pozemku H vlavo dole', 'property', 'Mensi rodinny domcek v zaujimavej lokalite.',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 410,110, 480,150 )
    )
);

INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
    46, 'Dom na pozemku S vlavo dole', 'property', 'Rodinny dom v pokojnej stvrti.',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 480,250, 540,280 )
    )
);

-- properties
INSERT INTO Properties VALUES (
    45, 'Ohnivaka 13', 'house', '1500000 CZK', 1
);

INSERT INTO Properties VALUES (
    46, 'Janosikova 4', 'house', '1800000 CZK', 2
);

-- block of flats
INSERT INTO SpatialEntities(id, name, type, geometry) VALUES (
    47, 'Bytovka na T', 'block of flats',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 610,340, 680,430 )
    )
);

INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
    48, 'Dom na pozemku X', 'property', 'Dom v lukrativnej stvrti.',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 710,450, 750,500 )
    )
);

INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
    49, 'Dom na pozemku P', 'property', 'Rodinny dom v pokojnej stvrti.',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 370,350, 420,400 )
    )
);

INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
    50, 'Dom na pozemku F', 'property', 'Dom v priemernej stvrti.',
    SDO_GEOMETRY(2003, NULL, NULL,
        SDO_ELEM_INFO_ARRAY(1, 1003, 3),
        SDO_ORDINATE_ARRAY( 50,110,100,220 )
    )
);

-- properties
INSERT INTO Properties VALUES (
    48, 'Necpalova 1', 'house', '1400000 CZK', 1
);

INSERT INTO Properties VALUES (
    49, 'Kramerova 44', 'house', '1900000 CZK', 2
);

INSERT INTO Properties VALUES (
    50, 'Hermanova 13', 'house', '2500000 CZK', 1
);





-- -- flat, !!! flat can be only inside the area of block of flats !!!
-- INSERT INTO SpatialEntities(id, name, type, description, geometry) VALUES (
--     48, 'Byt c.1 v bytovke na T', 'property', 'Krasny slnecny 3kk byt',
--     SDO_GEOMETRY(2001, NULL, SDO_POINT_TYPE(660,410, NULL), NULL,NULL)
-- );
-- -- flat as property
-- INSERT INTO Properties VALUES (
--     48, 'Dobrovskeho 14', 'flat', '1200000 CZK', 2
-- );

COMMIT;

SELECT name, id, SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(geometry, 0.1) valid FROM SpatialEntities;