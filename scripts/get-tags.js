cursor = db.derived_tag_concepts.find();
while ( cursor.hasNext() ) {
       printjson( cursor.next() );
}
