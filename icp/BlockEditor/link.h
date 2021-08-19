/**
* @file link.h
* @brief Link interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef ARROW_H
#define ARROW_H

#include <QGraphicsLineItem>

#include "schemeitem.h"
#include "connection.h"
#include "block.h"

class Link: public QGraphicsLineItem
{
public:
    enum { Type = UserType + 4 };
    Connection *connection;

    Link(SchemeItem *startItem, SchemeItem *endItem,QPointF endPoint, QGraphicsItem *parent = 0);
    SchemeItem *getSourceItem() const { return sourceItem; }
    SchemeItem *getDestItem() const { return destItem; }
    int type() const Q_DECL_OVERRIDE { return Type; }
    void updatePosition();

protected:
    QRectF boundingRect() const Q_DECL_OVERRIDE;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) Q_DECL_OVERRIDE;

private:
    QPointF sourcePort;
    QPointF destPort;
    SchemeItem *sourceItem;
    SchemeItem *destItem;
};

#endif // ARROW_H
