/**
* @file schemeitem.h
* @brief SchemeItem interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef DIAGRAMITEM_H
#define DIAGRAMITEM_H

#include <QGraphicsItem>
#include <QList>
#include "block.h"
#include "adder.h"
#include "subtractor.h"
#include "multiplier.h"
#include "divider.h"
#include "getimaginary.h"
#include "getreal.h"
#include "polarform.h"
#include "arithmblock.h"
#include "createcomplex.h"
#include "polartocomplex.h"

class Link;

class SchemeItem : public QGraphicsItem
{
public:
    enum DiagramType { Add, Sub, Mul, Div, GR, GI, GA, CC, CP,Step };
    enum { Type = UserType + 15 };

    Block *block;
    int input1;
    int input2;
    int output1;

    SchemeItem(DiagramType diagramType, QGraphicsItem *parent = 0);
    void removeLink(Link *link);
    void removeLinks();
    DiagramType diagramType() const { return itemType; }
    void addLink(Link *link);
    int type() const Q_DECL_OVERRIDE { return Type;}
    QRectF boundingRect() const Q_DECL_OVERRIDE;

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) Q_DECL_OVERRIDE;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) Q_DECL_OVERRIDE;

private:
    DiagramType itemType;
    QList<Link *> links;

};

#endif // DIAGRAMITEM_H
