/**
* @file link.cpp
* @brief Link implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "link.h"

#include <math.h>

#include <QPen>
#include <QPainter>

/**
 * @brief Link constructor
 * @param startItem Start item of connection
 * @param endItem End item of connection
 * @param endPoint Coordinates of destination port
 * @param parent
 */
Link::Link(SchemeItem *startItem, SchemeItem *endItem, QPointF endPoint ,QGraphicsItem *parent) : QGraphicsLineItem(parent)
{

    destPort = endPoint;
    sourcePort = QPointF(55,15);
    sourceItem = startItem;
    destItem = endItem;
    setFlag(QGraphicsItem::ItemIsSelectable, true);
}

/**
 * @brief Compute bounding rect of link
 * @return returns bounding rect
 */
QRectF Link::boundingRect() const
{
    qreal extra = (pen().width() + 20) / 2.0;

    return QRectF(line().p1(), QSizeF(line().p2().x() - line().p1().x(),
                                      line().p2().y() - line().p1().y()))
        .normalized()
        .adjusted(-extra, -extra, extra, extra);
}

/**
 * @brief Update position of link
 */
void Link::updatePosition()
{
    QLineF line(mapFromItem(sourceItem, sourcePort), mapFromItem(destItem, destPort));
    setLine(line);
}

/**
 * @brief Paints link between start and ent item. When startItem = endItem it does nothing.
 * @param painter
 */
void Link::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if (sourceItem->collidesWithItem(destItem))
        return;

    QPen myPen = pen();
    myPen.setWidth(3);
    painter->setPen(myPen);

    setLine(QLineF(mapFromItem(sourceItem, sourcePort), mapFromItem(destItem, destPort)));
    painter->drawLine(line());

    if (isSelected()) {

        painter->setPen(QPen(Qt::darkGray, 2, Qt::DashLine));
        QLineF myLine = line();
        myLine.translate(0, 4.0);
        painter->drawLine(myLine);
        myLine.translate(0,-8.0);
        painter->drawLine(myLine);
    }
}
