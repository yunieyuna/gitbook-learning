# Appendix A. ER Diagram for Example Database

## Appendix A. ER Diagram for Example Database

[Figure A-1](https://learning.oreilly.com/library/view/learning-sql-3rd/9781492057604/app01.html#learningsql-APP-A-FIG-1) is an entity-relationship (ER) diagram for the example database used in this book. As the name suggests, the diagram depicts the entities, or tables, in the database along with the foreign-key relationships between the tables. Here are a few tips to help you understand the notation:

* Each rectangle represents a table, with the table name above the upper-left corner of the rectangle. The primary-key column(s) are listed first, followed by nonkey columns.
* Lines between tables represent foreign key relationships. The markings at either end of the lines represent the allowable quantity, which can be zero (0), one (1), or many (<). For example, if you look at the relationship between the `customer` and `rental` tables, you would say that a rental is associated with exactly one customer, but a customer may have zero, one, or many rentals.

For more information on entity-relationship modeling, please see [the Wikipedia entry on this topic](https://oreil.ly/hLEeq).

<figure><img src="https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492057604/files/assets/lsq3_aa01.png" alt="lsql a01" height="1674" width="1416"><figcaption></figcaption></figure>

**Figure A-1. ER diagram**
