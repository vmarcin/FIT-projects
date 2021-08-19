/**
* @file schemescene.h
* @brief SchemeScene interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef DIAGRAMSCENE_H
#define DIAGRAMSCENE_H

#include "schemeitem.h"
#include "scheme.h"

#include <QGraphicsScene>

class SchemeScene : public QGraphicsScene
{
    Q_OBJECT

public:
    enum Mode { InsertItem, InsertLine, MoveItem };

    Scheme *scheme;
    explicit SchemeScene(QObject *parent = 0);

    void updateToolTips();
    void clearAll();
    void updatePositions();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent) Q_DECL_OVERRIDE;

private:
    Mode sceneMode;
    SchemeItem::DiagramType sceneItemType;
    QGraphicsLineItem *line;

public slots:
    void setMode(Mode sceneMode);
    void setItemType(SchemeItem::DiagramType type);

signals:
    void itemInserted(SchemeItem *item);
    void cycleError();
    void compatibleError();
};

#endif // DIAGRAMSCENE_H
